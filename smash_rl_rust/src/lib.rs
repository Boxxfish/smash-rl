pub mod micro_fighter_env;

use micro_fighter_env::*;
use pyo3::prelude::*;

#[pymodule]
fn smash_rl_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MicroFighterEnv>()?;
    Ok(())
}
