pub mod micro_fighter;
pub mod env;
mod move_states;
mod character;
mod hit;
mod ml;
mod training;

use micro_fighter::*;
use ml::GameState;
use pyo3::prelude::*;
use training::{test_jit, RolloutContext};

#[pymodule]
fn smash_rl_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MicroFighter>()?;
    m.add_class::<StepOutput>()?;
    m.add_class::<GameState>()?;
    m.add_class::<RolloutContext>()?;
    m.add_function(wrap_pyfunction!(test_jit, m)?)?;
    Ok(())
}
