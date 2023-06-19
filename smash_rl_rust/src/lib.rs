pub mod micro_fighter;
mod move_states;
mod character;
mod hit;
mod ml;

use micro_fighter::*;
use pyo3::prelude::*;

#[pymodule]
fn smash_rl_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MicroFighter>()?;
    Ok(())
}
