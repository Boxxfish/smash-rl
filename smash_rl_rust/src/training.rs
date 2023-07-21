use pyo3::prelude::*;
use tch::Tensor;

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

    pub fn insert_final_step(self, states_1: &Tensor, states_2: &Tensor) {
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
        let mut obs_1_vec = Vec::with_capacity(self.num_envs);
        let mut obs_2_vec = Vec::with_capacity(self.num_envs);
        let mut rewards = Vec::with_capacity(self.num_envs);
        let mut dones = Vec::with_capacity(self.num_envs);
        let mut truncs = Vec::with_capacity(self.num_envs);
        let mut infos = Vec::with_capacity(self.num_envs);

        for (i, action) in actions.iter().enumerate() {
            let ((obs_1, obs_2), reward, done, trunc, info) = self.envs[i].step(*action);
            obs_1_vec.push(obs_1);
            obs_2_vec.push(obs_2);
            rewards.push(reward);
            dones.push(done);
            truncs.push(trunc);
            infos.push(info);
        }

        let observations = (Tensor::stack(&obs_1_vec, 0), Tensor::stack(&obs_2_vec, 0));
        (observations, rewards, dones, truncs, infos)
    }

    pub fn reset(&mut self) -> (Tensor, Tensor) {
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
#[pyclass]
pub struct WorkerContext {
    last_obs_1: Tensor,
    last_obs_2: Tensor,
    rollout_buffer: RolloutBuffer,
    env: VecEnv,
    bot_ids: Vec<usize>,
}

#[pymethods]
impl WorkerContext {
    #[new]
    pub fn new(num_envs: usize, num_steps: u32, max_skip_frames: u32, num_frames: u32) -> Self {
        let mut env = VecEnv::new(
            (0..num_envs)
                .map(|_| MFEnv::new(max_skip_frames, num_frames, false, (0, 0, 0)))
                .collect(),
        );
        let (last_obs_1, last_obs_2) = env.reset();
        let (state_shape_1, state_shape_2) = env.envs[0].observation_space;
        let action_count = env.envs[0].action_space as i64;
        let rollout_buffer = RolloutBuffer::new(&state_shape_1, &state_shape_2, &[1], &[action_count], tch::Kind::Int, num_envs as u32, num_steps);
        let bot_ids = vec![0; num_envs];
        Self {
            last_obs_1,
            last_obs_2,
            rollout_buffer,
            env,
            bot_ids,
        }
    }
}
