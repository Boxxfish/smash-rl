use pyo3::prelude::*;
use tch;

/// Loads a model using JIT.
#[pyfunction]
pub fn test_jit() -> PyResult<()> {
    let model = tch::CModule::load("temp/training/p_net_test.ptc").expect("Couldn't load module.");
    Ok(())
}