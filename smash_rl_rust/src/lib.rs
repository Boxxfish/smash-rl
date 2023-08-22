pub mod micro_fighter;
pub mod env;
mod move_states;
mod character;
mod hit;
mod ml;
pub mod training;
pub mod melee;

use melee::{Gamepad, Button, Stick};
use micro_fighter::*;
use ml::GameState;
use pyo3::prelude::*;
use training::{test_jit, RolloutContext, BotData};

#[pymodule]
fn smash_rl_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MicroFighter>()?;
    m.add_class::<StepOutput>()?;
    m.add_class::<GameState>()?;
    m.add_class::<RolloutContext>()?;
    m.add_class::<BotData>()?;
    m.add_class::<Gamepad>()?;
    m.add_class::<Button>()?;
    m.add_class::<Stick>()?;
    m.add_function(wrap_pyfunction!(test_jit, m)?)?;
    Ok(())
}
