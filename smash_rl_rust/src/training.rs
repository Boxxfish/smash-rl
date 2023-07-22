use std::sync::{Arc, RwLock};

use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use rand::Rng;
use tch::{nn::Module, Tensor};
use weighted_rand::builder::{NewBuilder, WalkerTableBuilder};

use crate::env::{MFEnv, MFEnvInfo};

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
    states_1: Tensor,
    states_2: Tensor,
    actions: Tensor,
    action_probs: Tensor,
    rewards: Tensor,
    dones: Tensor,
    truncs: Tensor,
}

impl RolloutBuffer {
    pub fn new(
        state_shape_1: Size,
        state_shape_2: Size,
        action_shape: Size,
        action_probs_shape: Size,
        action_dtype: tch::Kind,
        num_envs: u32,
        num_steps: u32,
    ) -> Self {
        let k = tch::Kind::Float;
        let d = tch::Device::Cpu;
        let options = (k, d);
        let state_shape_1 = [&[num_steps as i64 + 1, num_envs as i64], state_shape_1].concat();
        let state_shape_2 = [&[num_steps as i64 + 1, num_envs as i64], state_shape_2].concat();
        let action_shape = [&[num_steps as i64, num_envs as i64], action_shape].concat();
        let action_probs_shape =
            [&[num_steps as i64, num_envs as i64], action_probs_shape].concat();
        let next = 0;
        let states_1 = Tensor::zeros(state_shape_1, options).set_requires_grad(false);
        let states_2 = Tensor::zeros(state_shape_2, options).set_requires_grad(false);
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
            states_1,
            states_2,
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
        states_1: &Tensor,
        states_2: &Tensor,
        actions: &Tensor,
        action_probs: &Tensor,
        rewards: &[f32],
        dones: &[bool],
        truncs: &[bool],
    ) {
        let d = tch::Device::Cpu;
        let _guard = tch::no_grad_guard();
        self.states_1.get(self.next).copy_(states_1);
        self.states_2.get(self.next).copy_(states_2);
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

    pub fn insert_final_step(&mut self, states_1: &Tensor, states_2: &Tensor) {
        let _guard = tch::no_grad_guard();
        self.states_1.get(self.next).copy_(states_1);
        self.states_2.get(self.next).copy_(states_2);
    }
}

/// Wrapper for environments for vectorization.
/// Only supports MFEnv.
pub struct VecEnv {
    pub envs: Vec<MFEnv>,
    pub num_envs: usize,
}

type VecEnvOutput = (
    (Tensor, Tensor),
    Vec<f32>,
    Vec<bool>,
    Vec<bool>,
    Vec<MFEnvInfo>,
);

impl VecEnv {
    pub fn new(envs: Vec<MFEnv>) -> Self {
        Self {
            num_envs: envs.len(),
            envs,
        }
    }

    pub fn step(&mut self, actions: &[u32]) -> VecEnvOutput {
        let _guard = tch::no_grad_guard();
        let mut obs_1_vec = Vec::with_capacity(self.num_envs);
        let mut obs_2_vec = Vec::with_capacity(self.num_envs);
        let mut rewards = Vec::with_capacity(self.num_envs);
        let mut dones = Vec::with_capacity(self.num_envs);
        let mut truncs = Vec::with_capacity(self.num_envs);
        let mut infos = Vec::with_capacity(self.num_envs);

        for (i, action) in actions.iter().enumerate() {
            let ((obs_1, obs_2), reward, done, trunc, info) = self.envs[i].step(*action);
            rewards.push(reward);
            dones.push(done);
            truncs.push(trunc);
            infos.push(info);

            // Reset if done or truncated
            if done || trunc {
                let ((obs_1, obs_2), _) = self.envs[i].reset();
                obs_1_vec.push(obs_1);
                obs_2_vec.push(obs_2);
            } else {
                obs_1_vec.push(obs_1);
                obs_2_vec.push(obs_2);
            }
        }

        let observations = (Tensor::stack(&obs_1_vec, 0), Tensor::stack(&obs_2_vec, 0));
        (observations, rewards, dones, truncs, infos)
    }

    pub fn reset(&mut self) -> (Tensor, Tensor) {
        let _guard = tch::no_grad_guard();
        let mut obs_1_vec = Vec::with_capacity(self.num_envs);
        let mut obs_2_vec = Vec::with_capacity(self.num_envs);

        for i in 0..self.num_envs {
            let ((obs_1, obs_2), _) = self.envs[i].reset();
            obs_1_vec.push(obs_1);
            obs_2_vec.push(obs_2);
        }

        (Tensor::stack(&obs_1_vec, 0), Tensor::stack(&obs_2_vec, 0))
    }
}

/// Stores worker state between iterations.
pub struct WorkerContext {
    last_obs_1: Tensor,
    last_obs_2: Tensor,
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
    ) -> Self {
        let mut env = VecEnv::new(
            (0..num_envs)
                .map(|_| MFEnv::new(max_skip_frames, num_frames, false, (0, 0, 0), time_limit))
                .collect(),
        );
        let (last_obs_1, last_obs_2) = env.reset();
        let (state_shape_1, state_shape_2) = env.envs[0].observation_space;
        let action_count = env.envs[0].action_space as i64;
        let rollout_buffer = RolloutBuffer::new(
            &state_shape_1,
            &state_shape_2,
            &[1],
            &[action_count],
            tch::Kind::Int,
            num_envs as u32,
            num_steps,
        );
        let bot_ids = vec![0; num_envs];
        Self {
            last_obs_1,
            last_obs_2,
            rollout_buffer,
            env,
            bot_ids,
            num_steps,
        }
    }
}

/// Stores data required for the entire rollout process.
#[pyclass]
pub struct RolloutContext {
    w_ctxs: Vec<WorkerContext>,
    bot_nets: Vec<tch::CModule>,
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
    ) -> Self {
        let envs_per_worker = total_num_envs / num_workers;
        let w_ctxs = (0..num_workers)
            .map(|_| {
                WorkerContext::new(
                    envs_per_worker,
                    num_steps,
                    max_skip_frames,
                    num_frames,
                    time_limit,
                )
            })
            .collect();
        let bot_nets = vec![tch::CModule::load(first_bot_path).expect("Couldn't load module.")];
        Self { w_ctxs, bot_nets }
    }

    /// Performs one iteration of the rollout process.
    /// Returns buffer tensors and entropy.
    pub fn rollout(
        &mut self,
        latest_policy_path: &str,
    ) -> (
        PyTensor,
        PyTensor,
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
        let mut bot_nets = Arc::new(RwLock::new(bot_nets));

        let mut handles = Vec::new();
        for mut w_ctx in self.w_ctxs.drain(0..self.w_ctxs.len()) {
            let p_net = p_net.clone();
            let bot_nets = bot_nets.clone();
            let handle = std::thread::spawn(move || {
                let _guard = tch::no_grad_guard();
                let mut rng = rand::thread_rng();
                let mut obs_1 = w_ctx.last_obs_1.copy();
                let mut obs_2 = w_ctx.last_obs_2.copy();
                let mut total_entropy = 0.0;
                for _ in 0..w_ctx.num_steps {
                    // Choose bot action
                    for (env_index, env_) in w_ctx.env.envs.iter_mut().enumerate() {
                        let (bot_obs_1, bot_obs_2) = env_.bot_obs();
                        let bot_action_probs = bot_nets.read().unwrap()[w_ctx.bot_ids[env_index]]
                            .forward_ts(&[bot_obs_1.unsqueeze(0), bot_obs_2.unsqueeze(0)])
                            .unwrap();
                        let bot_action = sample(&bot_action_probs)[0];
                        env_.bot_step(bot_action);
                    }

                    // Choose player action
                    let action_probs = p_net.forward_ts(&[&obs_1, &obs_2]).unwrap();
                    let actions = sample(&action_probs);

                    // Compute entropy.
                    // Based on the official Torch Distributions source.
                    let logits = &action_probs;
                    let probs = action_probs.softmax(-1, tch::Kind::Float);
                    let p_log_p = logits * probs;
                    let entropy = -p_log_p.sum_dim_intlist(-1, false, tch::Kind::Float);
                    total_entropy += entropy.mean(tch::Kind::Float).double_value(&[]);

                    let ((obs_1_, obs_2_), rewards, dones, truncs, _) = w_ctx.env.step(&actions);
                    obs_1 = obs_1_;
                    obs_2 = obs_2_;
                    w_ctx.rollout_buffer.insert_step(
                        &obs_1,
                        &obs_2,
                        &Tensor::from_slice(&actions.iter().map(|a| *a as i64).collect::<Vec<_>>())
                            .unsqueeze(1),
                        &action_probs,
                        &rewards,
                        &dones,
                        &truncs,
                    );

                    // Change opponent when environment ends
                    for (env_index, env_) in w_ctx.env.envs.iter().enumerate() {
                        if dones[env_index] || truncs[env_index] {
                            w_ctx.bot_ids[env_index] =
                                rng.gen_range(0..bot_nets.read().unwrap().len());
                        }
                    }
                }

                w_ctx.rollout_buffer.insert_final_step(&obs_1, &obs_2);
                w_ctx.last_obs_1 = obs_1;
                w_ctx.last_obs_2 = obs_2;

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
        let state_buffer_1 = Tensor::concatenate(
            &self
                .w_ctxs
                .iter()
                .map(|w_ctx| &w_ctx.rollout_buffer.states_1)
                .collect::<Vec<&Tensor>>(),
            1,
        );
        let state_buffer_2 = Tensor::concatenate(
            &self
                .w_ctxs
                .iter()
                .map(|w_ctx| &w_ctx.rollout_buffer.states_2)
                .collect::<Vec<&Tensor>>(),
            1,
        );
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
            PyTensor(state_buffer_1),
            PyTensor(state_buffer_2),
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
