use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use pyo3::prelude::*;

/// Stores all the plugins needed for human mode.
pub struct HumanPlugin;

impl Plugin for HumanPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Micro Fighter".into(),
                resolution: (400.0, 400.0).into(),
                ..default()
            }),
            ..default()
        }));
    }
}

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
            .add_plugin(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0))
            .add_plugin(RapierDebugRenderPlugin::default())
            .add_startup_system(setup)
            .run();
    }
}

impl Default for MicroFighterEnv {
    fn default() -> Self {
        Self::new()
    }
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
    commands.spawn((
        Collider::cuboid(500.0, 50.0),
        TransformBundle::from(Transform::from_xyz(0.0, -100.0, 0.0)),
    ));
    commands.spawn((
        RigidBody::Dynamic,
        Collider::ball(50.0),
        Restitution::coefficient(0.7),
        TransformBundle::from(Transform::from_xyz(0.0, 400.0, 0.0)),
    ));
}
