use std::sync::{Arc, RwLock};

use hora::core::ann_index::{ANNIndex, SerializableIndex};
use ndarray::prelude::*;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use rand::Rng;
use serde::Deserialize;
use tch::Tensor;
use weighted_rand::builder::{NewBuilder, WalkerTableBuilder};

use crate::env::{MFEnv, MFEnvInfo, IMG_SIZE};

/// Loads a model using JIT.
#[pyfunction]
pub fn test_jit() -> PyResult<()> {
    let _model = tch::CModule::load("temp/training/p_net_test.ptc").expect("Couldn't load module.");
    Ok(())
}

pub type Size<'a> = &'a [i64];

/// A pared down rollout buffer for collecting transitions.
/// The data should be sent to Python for actual training.
pub struct RolloutBuffer {
    num_envs: u32,
    num_steps: u32,
    next: i64,
    states: Vec<Tensor>,
    actions: Tensor,
    action_probs: Tensor,
    rewards: Tensor,
    dones: Tensor,
    truncs: Tensor,
}

impl RolloutBuffer {
    pub fn new(
        state_shapes: &[Size],
        action_shape: Size,
        action_probs_shape: Size,
        action_dtype: tch::Kind,
        num_envs: u32,
        num_steps: u32,
    ) -> Self {
        let k = tch::Kind::Float;
        let d = tch::Device::Cpu;
        let options = (k, d);
        let state_shapes: Vec<_> = state_shapes
            .iter()
            .map(|state_shape| [&[num_steps as i64 + 1, num_envs as i64], *state_shape].concat())
            .collect();
        let action_shape = [&[num_steps as i64, num_envs as i64], action_shape].concat();
        let action_probs_shape =
            [&[num_steps as i64, num_envs as i64], action_probs_shape].concat();
        let next = 0;
        let states = state_shapes
            .iter()
            .map(|state_shape| Tensor::zeros(state_shape, options).set_requires_grad(false))
            .collect();
        let actions = Tensor::zeros(action_shape, (action_dtype, d)).set_requires_grad(false);
        let action_probs = Tensor::zeros(action_probs_shape, options).set_requires_grad(false);
        let rewards =
            Tensor::zeros([num_steps as i64, num_envs as i64], options).set_requires_grad(false);
        let dones =
            Tensor::zeros([num_steps as i64, num_envs as i64], options).set_requires_grad(false);
        let truncs =
            Tensor::zeros([num_steps as i64, num_envs as i64], options).set_requires_grad(false);
        Self {
            num_envs,
            num_steps,
            next,
            states,
            actions,
            action_probs,
            rewards,
            dones,
            truncs,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert_step(
        &mut self,
        states: &[Tensor],
        actions: &Tensor,
        action_probs: &Tensor,
        rewards: &[f32],
        dones: &[bool],
        truncs: &[bool],
    ) {
        let _guard = tch::no_grad_guard();
        for (i, state) in states.iter().enumerate() {
            self.states[i].get(self.next).copy_(state);
        }
        self.actions.get(self.next).copy_(actions);
        self.action_probs.get(self.next).copy_(action_probs);
        self.rewards
            .get(self.next)
            .copy_(&Tensor::from_slice(rewards));
        self.dones.get(self.next).copy_(&Tensor::from_slice(dones));
        self.truncs
            .get(self.next)
            .copy_(&Tensor::from_slice(truncs));

        self.next += 1;
    }

    pub fn insert_final_step(&mut self, states: &[Tensor]) {
        let _guard = tch::no_grad_guard();
        for (i, state) in states.iter().enumerate() {
            self.states[i].get(self.next).copy_(state);
        }
    }
}

/// Wrapper for environments for vectorization.
/// Only supports MFEnv.
pub struct VecEnv {
    pub envs: Vec<MFEnv>,
    pub num_envs: usize,
}

type VecEnvOutput = (Vec<Tensor>, Vec<f32>, Vec<bool>, Vec<bool>, Vec<MFEnvInfo>);

impl VecEnv {
    pub fn new(envs: Vec<MFEnv>) -> Self {
        Self {
            num_envs: envs.len(),
            envs,
        }
    }

    pub fn step(&mut self, actions: &[u32]) -> VecEnvOutput {
        let _guard = tch::no_grad_guard();
        let mut obs_vec: Vec<_> = (0..2).map(|_| Vec::with_capacity(self.num_envs)).collect();
        let mut rewards = Vec::with_capacity(self.num_envs);
        let mut dones = Vec::with_capacity(self.num_envs);
        let mut truncs = Vec::with_capacity(self.num_envs);
        let mut infos = Vec::with_capacity(self.num_envs);

        for (i, action) in actions.iter().enumerate() {
            let (obs, reward, done, trunc, info) = self.envs[i].step(*action);
            rewards.push(reward);
            dones.push(done);
            truncs.push(trunc);
            infos.push(info);

            // Reset if done or truncated
            if done || trunc {
                let (obs, _) = self.envs[i].reset();
                for (i, obs) in obs.iter().enumerate() {
                    obs_vec[i].push(obs.copy());
                }
            } else {
                for (i, obs) in obs.iter().enumerate() {
                    obs_vec[i].push(obs.copy());
                }
            }
        }

        let observations = obs_vec
            .iter()
            .map(|obs_vec| Tensor::stack(obs_vec, 0))
            .collect();
        (observations, rewards, dones, truncs, infos)
    }

    pub fn reset(&mut self) -> Vec<Tensor> {
        let _guard = tch::no_grad_guard();
        let mut obs_vec: Vec<_> = (0..2).map(|_| Vec::with_capacity(self.num_envs)).collect();

        for i in 0..self.num_envs {
            let (obs, _) = self.envs[i].reset();
            for (i, obs) in obs.iter().enumerate() {
                obs_vec[i].push(obs.copy());
            }
        }

        obs_vec
            .iter()
            .map(|obs_vec| Tensor::stack(obs_vec, 0))
            .collect()
    }
}

/// Stores worker state between iterations.
pub struct WorkerContext {
    last_obs: Vec<Tensor>,
    rollout_buffer: RolloutBuffer,
    env: VecEnv,
    bot_ids: Vec<usize>,
    num_steps: u32,
}

impl WorkerContext {
    pub fn new(
        num_envs: usize,
        num_steps: u32,
        max_skip_frames: u32,
        num_frames: u32,
        time_limit: u32,
        top_k: u32,
        retrieval_ctx: &Arc<RwLock<RetrievalContext>>,
    ) -> Self {
        let mut env = VecEnv::new(
            (0..num_envs)
                .map(|_| MFEnv::new(max_skip_frames, num_frames, false, (0, 0, 0), time_limit))
                .collect(),
        );
        let last_obs = add_retrieval(env.reset(), top_k as usize, retrieval_ctx);
        let state_shape = &[
            env.envs[0].observation_space.as_slice(),
            vec![
                vec![top_k as i64, 16, IMG_SIZE as i64, IMG_SIZE as i64],
                vec![top_k as i64, 181],
            ]
            .as_slice(),
        ]
        .concat();
        let action_count = env.envs[0].action_space as i64;
        let rollout_buffer = RolloutBuffer::new(
            &state_shape.iter().map(|s| s.as_slice()).collect::<Vec<_>>(),
            &[1],
            &[action_count],
            tch::Kind::Int,
            num_envs as u32,
            num_steps,
        );
        let bot_ids = vec![0; num_envs];
        Self {
            last_obs,
            rollout_buffer,
            env,
            bot_ids,
            num_steps,
        }
    }
}

/// Stores metadata on each bot.
#[pyclass]
#[derive(Clone)]
pub struct BotData {
    #[pyo3(get)]
    pub elo: f32,
}

/// Stores data required for the entire rollout process.
#[pyclass]
pub struct RolloutContext {
    w_ctxs: Vec<WorkerContext>,
    bot_nets: Vec<tch::CModule>,
    bot_data: Vec<BotData>,
    retrieval_ctx: Arc<RwLock<RetrievalContext>>,
    top_k: u32,
    test_env: MFEnv,
    p_net: tch::CModule,
    current_elo: f32,
    initial_elo: f32,
}

#[pymethods]
impl RolloutContext {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        total_num_envs: usize,
        num_workers: usize,
        num_steps: u32,
        max_skip_frames: u32,
        num_frames: u32,
        time_limit: u32,
        first_bot_path: &str,
        top_k: u32,
        initial_elo: f32,
    ) -> Self {
        // Set up retrieval context
        let encoder = tch::jit::CModule::load("temp/encoder.ptc").unwrap();
        let pca = PCA::load("temp/pca.json");
        let retrieval_ctx = Arc::new(RwLock::new(RetrievalContext::new(
            128,
            "temp/index.bin",
            "temp/generated",
            "temp/episode_data.json",
            encoder,
            pca,
        )));

        // Create worker contexts
        let envs_per_worker = total_num_envs / num_workers;
        let w_ctxs = (0..num_workers)
            .map(|_| {
                WorkerContext::new(
                    envs_per_worker,
                    num_steps,
                    max_skip_frames,
                    num_frames,
                    time_limit,
                    top_k,
                    &retrieval_ctx,
                )
            })
            .collect();

        let p_net = tch::CModule::load(first_bot_path).expect("Couldn't load module.");
        let bot_nets = vec![tch::CModule::load(first_bot_path).expect("Couldn't load module.")];
        let bot_data = vec![BotData { elo: initial_elo }];
        let test_env = MFEnv::new(max_skip_frames, num_frames, false, (0, 0, 0), time_limit);

        Self {
            w_ctxs,
            bot_nets,
            retrieval_ctx,
            top_k,
            bot_data,
            test_env,
            p_net,
            current_elo: initial_elo,
            initial_elo,
        }
    }

    /// Performs one iteration of the rollout process.
    /// Returns buffer tensors and entropy.
    pub fn rollout(
        &mut self,
        latest_policy_path: &str,
    ) -> (
        Vec<PyTensor>,
        PyTensor,
        PyTensor,
        PyTensor,
        PyTensor,
        PyTensor,
        f32,
    ) {
        let _guard = tch::no_grad_guard();
        let p_net =
            Arc::new(tch::CModule::load(latest_policy_path).expect("Couldn't load module."));
        let mut bot_nets = Vec::new();
        bot_nets.append(&mut self.bot_nets);
        let bot_nets = Arc::new(RwLock::new(bot_nets));
        let top_k = self.top_k as usize;

        let mut handles = Vec::new();
        for mut w_ctx in self.w_ctxs.drain(0..self.w_ctxs.len()) {
            let p_net = p_net.clone();
            let bot_nets = bot_nets.clone();
            let retrieval_ctx = self.retrieval_ctx.clone();
            let handle = std::thread::spawn(move || {
                let _guard = tch::no_grad_guard();
                let mut rng = rand::thread_rng();
                let mut obs: Vec<_> = w_ctx.last_obs.iter().map(|t| t.copy()).collect();
                let mut total_entropy = 0.0;
                for _ in 0..w_ctx.num_steps {
                    // Choose bot action
                    for (env_index, env_) in w_ctx.env.envs.iter_mut().enumerate() {
                        let bot_obs: Vec<_> = env_
                            .bot_obs()
                            .iter()
                            .map(|t| t.to_kind(tch::Kind::Float).unsqueeze(0))
                            .collect();
                        let bot_obs = add_retrieval(bot_obs, top_k, &retrieval_ctx);
                        let bot_action_probs = bot_nets.read().unwrap()[w_ctx.bot_ids[env_index]]
                            .forward_ts(
                                &bot_obs
                                    .iter()
                                    .map(|t| t.to_kind(tch::Kind::Float))
                                    .collect::<Vec<_>>(),
                            )
                            .unwrap();
                        let bot_action = sample(&bot_action_probs)[0];
                        env_.bot_step(bot_action);
                    }

                    // Choose player action
                    obs = obs.iter().map(|o| o.to_kind(tch::Kind::Float)).collect();
                    let action_probs = p_net.forward_ts(&obs).unwrap();
                    let actions = sample(&action_probs);

                    // Compute entropy.
                    // Based on the official Torch Distributions source.
                    let logits = &action_probs;
                    let probs = action_probs.softmax(-1, tch::Kind::Float);
                    let p_log_p = logits * probs;
                    let entropy = -p_log_p.sum_dim_intlist(-1, false, tch::Kind::Float);
                    total_entropy += entropy.mean(tch::Kind::Float).double_value(&[]);

                    let (obs_, rewards, dones, truncs, _) = w_ctx.env.step(&actions);
                    obs = add_retrieval(obs_, top_k, &retrieval_ctx);
                    w_ctx.rollout_buffer.insert_step(
                        &obs,
                        &Tensor::from_slice(&actions.iter().map(|a| *a as i64).collect::<Vec<_>>())
                            .unsqueeze(1),
                        &action_probs,
                        &rewards,
                        &dones,
                        &truncs,
                    );

                    // Change opponent when environment ends
                    for env_index in 0..w_ctx.env.num_envs {
                        if dones[env_index] || truncs[env_index] {
                            w_ctx.bot_ids[env_index] =
                                rng.gen_range(0..bot_nets.read().unwrap().len());
                        }
                    }
                }

                w_ctx.rollout_buffer.insert_final_step(&obs);
                w_ctx.last_obs = obs;

                (w_ctx, total_entropy)
            });
            handles.push(handle);
        }

        // Process data once finished
        let mut total_entropy = 0.0;
        for handle in handles {
            let (w_ctx, w_total_entropy) = handle.join().unwrap();
            self.w_ctxs.push(w_ctx);
            total_entropy += w_total_entropy;
        }
        self.bot_nets.append(bot_nets.write().as_mut().unwrap());

        // Copy the contents of each rollout buffer
        let state_buffers = (0..self.w_ctxs[0].rollout_buffer.states.len())
            .map(|i| {
                PyTensor(Tensor::concatenate(
                    &self
                        .w_ctxs
                        .iter()
                        .map(|w_ctx| &w_ctx.rollout_buffer.states[i])
                        .collect::<Vec<&Tensor>>(),
                    1,
                ))
            })
            .collect();
        let act_buffer = Tensor::concatenate(
            &self
                .w_ctxs
                .iter()
                .map(|w_ctx| &w_ctx.rollout_buffer.actions)
                .collect::<Vec<&Tensor>>(),
            1,
        );
        let act_probs_buffer = Tensor::concatenate(
            &self
                .w_ctxs
                .iter()
                .map(|w_ctx| &w_ctx.rollout_buffer.action_probs)
                .collect::<Vec<&Tensor>>(),
            1,
        );
        let reward_buffer = Tensor::concatenate(
            &self
                .w_ctxs
                .iter()
                .map(|w_ctx| &w_ctx.rollout_buffer.rewards)
                .collect::<Vec<&Tensor>>(),
            1,
        );
        let done_buffer = Tensor::concatenate(
            &self
                .w_ctxs
                .iter()
                .map(|w_ctx| &w_ctx.rollout_buffer.dones)
                .collect::<Vec<&Tensor>>(),
            1,
        );
        let trunc_buffer = Tensor::concatenate(
            &self
                .w_ctxs
                .iter()
                .map(|w_ctx| &w_ctx.rollout_buffer.truncs)
                .collect::<Vec<&Tensor>>(),
            1,
        );
        for w_ctx in self.w_ctxs.iter_mut() {
            w_ctx.rollout_buffer.next = 0;
        }

        let entropy =
            total_entropy as f32 / (self.w_ctxs.len() * self.w_ctxs[0].num_steps as usize) as f32;

        (
            state_buffers,
            PyTensor(act_buffer),
            PyTensor(act_probs_buffer),
            PyTensor(reward_buffer),
            PyTensor(done_buffer),
            PyTensor(trunc_buffer),
            entropy,
        )
    }

    /// Appends a new bot.
    pub fn push_bot(&mut self, path: &str) {
        let bot_net = tch::CModule::load(path).expect("Couldn't load module.");
        self.bot_nets.push(bot_net);
    }

    /// Inserts a new bot at the given position.
    pub fn insert_bot(&mut self, path: &str, index: usize) {
        let bot_net = tch::CModule::load(path).expect("Couldn't load module.");
        self.bot_nets[index] = bot_net;
    }

    /// Sets the exploration reward amount.
    pub fn set_expl_reward_amount(&mut self, amount: f32) {
        for w_ctx in &mut self.w_ctxs {
            for env in &mut w_ctx.env.envs {
                env.set_dmg_reward_amount(amount);
            }
        }
    }

    /// Performs evaluation against bots and updates ELO.
    pub fn perform_eval(&mut self, eval_steps: usize, max_eval_steps: usize, elo_k: u32) {
        let _guard = tch::no_grad_guard();
        let (obs_, _) = self.test_env.reset();
        let mut obs = add_retrieval(
            obs_.iter()
                .map(|o| o.to_kind(tch::Kind::Float).unsqueeze(0))
                .collect::<Vec<_>>(),
            self.top_k as usize,
            &self.retrieval_ctx,
        )
        .iter()
        .map(|t| t.to_kind(tch::Kind::Float))
        .collect::<Vec<_>>();
        let mut rng = rand::thread_rng();
        for _ in 0..eval_steps {
            let eval_bot_index = rng.gen_range(0..self.bot_nets.len());
            let b_elo = self.bot_data[eval_bot_index].elo;
            for _ in 0..max_eval_steps {
                let bot_obs = add_retrieval(
                    self.test_env
                        .bot_obs()
                        .iter()
                        .map(|o| o.to_kind(tch::Kind::Float).unsqueeze(0))
                        .collect::<Vec<_>>(),
                    self.top_k as usize,
                    &self.retrieval_ctx,
                )
                .iter()
                .map(|t| t.to_kind(tch::Kind::Float))
                .collect::<Vec<_>>();
                let bot_action_probs = self.bot_nets[eval_bot_index].forward_ts(&bot_obs).unwrap();
                let bot_action = sample(&bot_action_probs)[0];
                self.test_env.bot_step(bot_action);

                let action_probs = self.p_net.forward_ts(&obs).unwrap();
                let action = sample(&action_probs)[0];
                let (obs_, _, done, trunc, info) = self.test_env.step(action);
                obs = add_retrieval(
                    obs_.iter()
                        .map(|o| o.to_kind(tch::Kind::Float).unsqueeze(0))
                        .collect::<Vec<_>>(),
                    self.top_k as usize,
                    &self.retrieval_ctx,
                )
                .iter()
                .map(|t| t.to_kind(tch::Kind::Float))
                .collect::<Vec<_>>();
                if done || trunc {
                    let (a, b) = if done {
                        if info.player_won {
                            // Current network won
                            (1.0, 0.0)
                        } else {
                            // Opponent won
                            (0.0, 1.0)
                        }
                    } else {
                        // They tied
                        (0.5, 0.5)
                    };
                    let ea = 1.0 / (1.0 + 10.0_f32.powf((b_elo - self.current_elo) / 400.0));
                    let eb = 1.0 / (1.0 + 10.0_f32.powf((self.current_elo - b_elo) / 400.0));
                    self.current_elo += elo_k as f32 * (a - ea);
                    self.bot_data[eval_bot_index].elo = b_elo + elo_k as f32 * (b - eb);
                    let (obs_, _) = self.test_env.reset();
                    obs = add_retrieval(
                        obs_.iter()
                            .map(|o| o.to_kind(tch::Kind::Float).unsqueeze(0))
                            .collect::<Vec<_>>(),
                        self.top_k as usize,
                        &self.retrieval_ctx,
                    )
                    .iter()
                    .map(|t| t.to_kind(tch::Kind::Float))
                    .collect::<Vec<_>>();
                    break;
                }
            }
        }
    }

    /// Returns the current ELO.
    pub fn current_elo(&self) -> f32 {
        self.current_elo
    }

    /// Returns a list of bot data.
    pub fn bot_data(&self) -> Vec<BotData> {
        self.bot_data.clone()
    }
}

/// Adds retrieval info from an MFEnv observation.
fn add_retrieval(
    mut orig_obs: Vec<Tensor>,
    top_k: usize,
    retrieval_ctx: &Arc<RwLock<RetrievalContext>>,
) -> Vec<Tensor> {
    let n_obs = retrieval_ctx
        .read()
        .unwrap()
        .search_from_obs(&orig_obs[0], &orig_obs[1], top_k);
    orig_obs.push(n_obs.0);
    orig_obs.push(n_obs.1);
    orig_obs
}

/// Returns a list of actions given the probabilities.
fn sample(logits: &Tensor) -> Vec<u32> {
    let num_samples = logits.size()[0];
    let num_weights = logits.size()[1] as usize;
    let mut generated_samples = Vec::with_capacity(num_samples as usize);
    let mut rng = rand::thread_rng();
    let probs = logits.softmax(-1, tch::Kind::Float);

    for i in 0..num_samples {
        let mut weights = vec![0.0; num_weights];
        probs.get(i).copy_data(&mut weights, num_weights);
        let builder = WalkerTableBuilder::new(&weights);
        let table = builder.build();
        let action = table.next_rng(&mut rng);
        generated_samples.push(action as u32);
    }

    generated_samples
}

#[derive(Deserialize)]
struct PCAData {
    basis: Vec<Vec<f32>>,
    mean: Vec<f32>,
}

/// Performs PCA.
pub struct PCA {
    basis: Array2<f32>,
    mean: Array1<f32>,
}

impl PCA {
    /// Loads PCA data from a file.
    pub fn load(path: &str) -> Self {
        let file = std::fs::File::open(path).unwrap();
        let reader = std::io::BufReader::new(file);
        let data: PCAData = serde_json::from_reader(reader).unwrap();
        let basis = data
            .basis
            .iter()
            .map(|b| arr1(&b.clone()))
            .collect::<Vec<_>>();
        let basis =
            ndarray::stack(Axis(0), &basis.iter().map(|b| b.view()).collect::<Vec<_>>()).unwrap();
        let mean = arr1(&data.mean);
        Self { basis, mean }
    }

    /// Given a tensor of (batch_size, a), returns (batch_size, b).
    /// Based on the Scikit-learn implementation.
    pub fn transform(&self, data: &Tensor) -> Tensor {
        (data
            - Tensor::try_from(&self.mean)
                .unwrap()
                .unsqueeze(0)
                .repeat([data.size()[0], 1]))
        .matmul(&Tensor::try_from(&self.basis).unwrap().t_copy())
    }
}

#[derive(Deserialize)]
struct EpisodeData {
    pub player_won: f32,
    pub traj_in_episode: Vec<usize>,
}

/// Central location for data needed for retrieval.
pub struct RetrievalContext {
    encoder: tch::CModule,
    pca: PCA,
    index: hora::index::hnsw_idx::HNSWIndex<f32, usize>,
    key_dim: usize,
    data_spatial: ArrayD<f64>,
    data_scalar: ArrayD<f64>,
}

impl RetrievalContext {
    pub fn new(
        key_dim: usize,
        index_path: &str,
        generated_dir: &str,
        episode_data_path: &str,
        encoder: tch::CModule,
        pca: PCA,
    ) -> Self {
        // Load index
        let mut index = hora::index::hnsw_idx::HNSWIndex::<f32, usize>::load(index_path).unwrap();
        index
            .build(hora::core::metrics::Metric::DotProduct)
            .unwrap();

        // Set up episode data
        let mut traj_data = Vec::new();
        let file = std::fs::File::open(episode_data_path).unwrap();
        let reader = std::io::BufReader::new(file);
        let episode_data: Vec<EpisodeData> = serde_json::from_reader(reader).unwrap();
        for data in episode_data {
            let traj_in_episode = data.traj_in_episode.len();
            for _ in 0..traj_in_episode {
                traj_data.push(data.player_won);
            }
        }

        // Load trajectory data
        let data_path = std::fs::read_dir(generated_dir).unwrap();
        let file_count = data_path.count() / 2;
        let mut data_spatial = Vec::with_capacity(file_count);
        let mut data_scalar = Vec::with_capacity(file_count);
        for i in 0..file_count {
            data_spatial.push(load_npy_as_tensor(&format!(
                "{generated_dir}/{i}_data_spatial.npy"
            )));
            data_scalar.push(load_npy_as_tensor(&format!(
                "{generated_dir}/{i}_data_scalar.npy"
            )));
        }
        let data_spatial = Tensor::concatenate(&data_spatial, 0)
            .as_ref()
            .try_into()
            .unwrap();
        let data_scalar = Tensor::concatenate(&data_scalar, 0);
        let extra_scalar = Tensor::zeros(
            [data_scalar.size()[0], 1],
            (tch::Kind::Float, tch::Device::Cpu),
        );
        for (i, &player_won) in traj_data.iter().enumerate() {
            extra_scalar.get(i as i64).get(0).set_data(
                &(Tensor::scalar_tensor(player_won as f64, (tch::Kind::Float, tch::Device::Cpu))),
            );
        }
        let data_scalar: ArrayD<f64> = Tensor::concatenate(&[data_scalar, extra_scalar], 1)
            .as_ref()
            .try_into()
            .unwrap();

        Self {
            encoder,
            pca,
            index,
            key_dim,
            data_spatial,
            data_scalar,
        }
    }

    pub fn search(&self, key: &[f32], top_k: usize) -> (Tensor, Tensor) {
        let indices = &self.index.search(key, top_k);
        let results_spatial = self.data_spatial.select(Axis(0), indices);
        let results_scalar = self.data_scalar.select(Axis(0), indices);
        (
            results_spatial.try_into().unwrap(),
            results_scalar.try_into().unwrap(),
        )
    }

    /// Performs batch search from observations.
    pub fn search_from_obs(
        &self,
        spatial: &Tensor,
        stats: &Tensor,
        top_k: usize,
    ) -> (Tensor, Tensor) {
        let _guard = tch::no_grad_guard();
        let encoded = self
            .encoder
            .forward_ts(&[
                spatial.to_kind(tch::Kind::Float),
                stats.to_kind(tch::Kind::Float),
            ])
            .unwrap();
        let keys = self.pca.transform(&encoded);
        let key_count = keys.size()[0] as usize;
        let mut n_spatials = Vec::with_capacity(key_count);
        let mut n_scalars = Vec::with_capacity(key_count);
        let mut keys_data = vec![0.0; self.key_dim * key_count];
        keys.copy_data(&mut keys_data, self.key_dim * key_count);
        for i in 0..key_count {
            let (n_spatial, n_scalar) = self.search(
                &keys_data[i * self.key_dim..((i + 1) * self.key_dim)],
                top_k,
            );
            n_spatials.push(n_spatial);
            n_scalars.push(n_scalar);
        }
        (Tensor::stack(&n_spatials, 0), Tensor::stack(&n_scalars, 0))
    }
}

/// Loads numpy data from a path.
fn load_npy_as_tensor(path: &str) -> Tensor {
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);
    let np_file = npyz::NpyFile::new(reader).unwrap();
    let shape = np_file.shape().to_vec();
    let data: Vec<f64> = np_file.into_vec().unwrap();
    Tensor::from_slice(&data).reshape(shape.iter().map(|e| *e as i64).collect::<Vec<_>>())
}
