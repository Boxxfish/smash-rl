pub mod micro_fighter_env;
mod move_states;
mod character;
mod hit;

use micro_fighter_env::*;
use pyo3::prelude::*;

#[pymodule]
fn smash_rl_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MicroFighterEnv>()?;
    Ok(())
}
