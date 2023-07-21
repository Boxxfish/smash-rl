use pyo3::prelude::*;
use tch::Tensor;

/// Loads a model using JIT.
#[pyfunction]
pub fn test_jit() -> PyResult<()> {
    let model = tch::CModule::load("temp/training/p_net_test.ptc").expect("Couldn't load module.");
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
        self.rewards.get(self.next).copy_(&Tensor::from_slice(rewards));
        self.dones.get(self.next).copy_(&Tensor::from_slice(dones));
        self.truncs.get(self.next).copy_(&Tensor::from_slice(truncs));

        self.next += 1;
    }

    pub fn insert_final_step(self, states_1: &Tensor, states_2: &Tensor) {
        let _guard = tch::no_grad_guard();
        self.states_1.get(self.next).copy_(states_1);
        self.states_2.get(self.next).copy_(states_2);
    }
}
