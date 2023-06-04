use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use pyo3::prelude::*;

use crate::character::*;
use crate::hit::*;
use crate::move_states::*;

pub const FIXED_TIMESTEP: f32 = 1.0 / 60.0;

// Collision groups:
// 0: Floor
// 1: Player floor collider
// 2: Opponent floor collider
pub const FLOOR_COLL_GROUP: u32 = 0b001;
pub const PLAYER_COLL_GROUP: u32 = 0b010;
pub const OPPONENT_COLL_GROUP: u32 = 0b100;
pub const FLOOR_COLL_FILTER: u32 = 0b111;
pub const PLAYER_COLL_FILTER: u32 = 0b001;
pub const OPPONENT_COLL_FILTER: u32 = 0b001;

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

/// Whether the game is loading.
#[derive(States, Debug, Hash, PartialEq, Eq, Clone, Default)]
pub enum AppState {
    #[default]
    Loading,
    Running,
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
            .add_plugin(CharacterPlugin)
            .add_plugin(MoveStatesPlugin)
            .add_plugin(HitPlugin)
            .add_state::<AppState>()
            .add_startup_system(setup)
            .run();
    }
}

impl Default for MicroFighterEnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Denotes a floor.
#[derive(Component)]
pub struct Floor;

fn setup(mut commands: Commands, mut app_state: ResMut<NextState<AppState>>) {
    commands.spawn(Camera2dBundle::default());
    // Floor
    commands.spawn((
        Floor,
        Collider::cuboid(500.0, 50.0),
        TransformBundle::from(Transform::from_xyz(0.0, -200.0, 0.0)),
        CollisionGroups::new(
            Group::from_bits(FLOOR_COLL_GROUP).unwrap(),
            Group::from_bits(FLOOR_COLL_FILTER).unwrap(),
        ),
    ));
    // Player
    let p_floor_collider = commands
        .spawn((
            Collider::cuboid(8.0, 4.0),
            TransformBundle::from(Transform::from_xyz(0.0, -30.0, 0.0)),
            ActiveEvents::COLLISION_EVENTS,
            CollisionGroups::new(
                Group::from_bits(PLAYER_COLL_GROUP).unwrap(),
                Group::from_bits(PLAYER_COLL_FILTER).unwrap(),
            ),
        ))
        .id();
    let mut p_bundle = CharBundle::default();
    p_bundle.character.floor_collider = Some(p_floor_collider);
    commands
        .spawn((Player::default(), p_bundle))
        .add_child(p_floor_collider);
    // Bot
    let b_floor_collider = commands
        .spawn((
            Collider::cuboid(8.0, 4.0),
            TransformBundle::from(Transform::from_xyz(0.0, -30.0, 0.0)),
            ActiveEvents::COLLISION_EVENTS,
            CollisionGroups::new(
                Group::from_bits(OPPONENT_COLL_GROUP).unwrap(),
                Group::from_bits(OPPONENT_COLL_FILTER).unwrap(),
            ),
        ))
        .id();
    let mut b_bundle = CharBundle {
        transform: TransformBundle::from(Transform::from_xyz(50.0, 0.0, 0.0)),
        ..default()
    };
    b_bundle.character.floor_collider = Some(b_floor_collider);
    commands.spawn((Bot, b_bundle)).add_child(b_floor_collider);
    app_state.set(AppState::Running);
}
