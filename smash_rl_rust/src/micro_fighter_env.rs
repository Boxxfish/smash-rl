use bevy::prelude::*;
use pyo3::prelude::*;

/// Simple fighting game environment.
#[pyclass]
pub struct MicroFighterEnv {}

#[pymethods]
impl MicroFighterEnv {
    #[new]
    pub fn new() -> MicroFighterEnv {
        MicroFighterEnv {}
    }

    /// Runs the environment in human mode.
    pub fn run(&self) {
        App::new()
            .add_plugin(HumanPlugin)
            .add_system(hello_world)
            .run();
    }
}

impl Default for MicroFighterEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Stores all the plugins needed for human mode.
pub struct HumanPlugin;

impl Plugin for HumanPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(DefaultPlugins);
    }
}

fn hello_world() {
    println!("Hello World!");
}
