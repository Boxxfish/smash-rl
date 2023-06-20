use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use pyo3::prelude::*;

use crate::character::*;
use crate::hit::*;
use crate::ml::HBox;
use crate::ml::HBoxCollection;
use crate::ml::MLPlayerActionEvent;
use crate::ml::MLPlugin;
use crate::move_states::*;

pub const FIXED_TIMESTEP: f32 = 1.0 / 60.0;
pub const SCREEN_SIZE: u32 = 400;

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
                resolution: (SCREEN_SIZE as f32, SCREEN_SIZE as f32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugin(RapierDebugRenderPlugin::default())
        .add_system(player_input.in_set(OnUpdate(AppState::Running)));
    }
}

/// Whether the game is loading.
#[derive(States, Debug, Hash, PartialEq, Eq, Clone, Default)]
pub enum AppState {
    #[default]
    Loading,
    Running,
}

/// Stores output of a step.
#[pyclass]
pub struct StepOutput {
    /// HBoxes in the scene.
    #[pyo3(get)]
    pub hboxes: Vec<HBox>,
    /// Whether the round is over.
    #[pyo3(get)]
    pub round_over: bool,
    /// Whether the player won.
    #[pyo3(get)]
    pub player_won: bool,
}

/// Simple fighting game.
#[pyclass]
pub struct MicroFighter {
    app: App,
    first_step: bool,
}

#[pymethods]
impl MicroFighter {
    #[new]
    /// Creates a new MicroFighter environment.
    ///
    /// * `human`: If true, the environment acts as a playable game. Otherwise,
    ///   acts as an RL environment.
    pub fn new(human: bool) -> MicroFighter {
        let mut app = App::new();
        if human {
            app.add_plugin(HumanPlugin);
        } else {
            app.add_plugin(MLPlugin);
        }
        app.add_plugin(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0))
            .add_plugin(CharacterPlugin)
            .add_plugin(MoveStatesPlugin)
            .add_plugin(HitPlugin)
            .add_state::<AppState>()
            .add_event::<ResetEvent>()
            .add_system(handle_reset)
            .add_startup_system(setup);
        MicroFighter {
            app,
            first_step: true,
        }
    }

    /// Runs the environment.
    /// The environment must be in human mode.
    pub fn run(&mut self) {
        self.first_step = false;
        self.app.run();
    }

    /// Runs one step of the environment and returns hitbox and hurtbox info.
    /// The environment must not be in human mode.
    pub fn step(&mut self, action_id: u32) -> StepOutput {
        if self.first_step {
            self.app.setup();
            self.first_step = false;
        }
        self.app.world.send_event(MLPlayerActionEvent { action_id });
        self.app.update();
        let world = &self.app.world;

        let events = world.get_resource::<Events<RoundOverEvent>>().unwrap();
        let mut ev_round_over = events.get_reader();
        let (round_over, player_won) = if let Some(ev) = ev_round_over.iter(events).next() {
            let player_won = ev.player_won;
            (true, player_won)
        } else {
            (false, false)
        };

        let hbox_coll = world.get_resource::<HBoxCollection>().unwrap();
        StepOutput {
            hboxes: hbox_coll.hboxes.clone(),
            round_over,
            player_won,
        }
    }

    /// Resets the environment.
    pub fn reset(&mut self) -> StepOutput {
        if self.first_step {
            self.app.setup();
            self.first_step = false;
        }
        self.app.world.send_event(ResetEvent);
        self.app.update();
        let world = &self.app.world;

        let events = world.get_resource::<Events<RoundOverEvent>>().unwrap();
        let mut ev_round_over = events.get_reader();
        let (round_over, player_won) = if let Some(ev) = ev_round_over.iter(events).next() {
            let player_won = ev.player_won;
            (true, player_won)
        } else {
            (false, false)
        };

        let hbox_coll = world.get_resource::<HBoxCollection>().unwrap();
        StepOutput {
            hboxes: hbox_coll.hboxes.clone(),
            round_over,
            player_won,
        }
    }

    /// Returns the internal screen size.
    pub fn get_screen_size(&self) -> u32 {
        SCREEN_SIZE
    }
}

/// Denotes a floor.
#[derive(Component)]
pub struct Floor;

fn setup(mut commands: Commands, mut ev_reset: EventWriter<ResetEvent>) {
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

    // All dynamic stuff gets handled by resetting
    ev_reset.send(ResetEvent);
}

/// Handles input from the keyboard.
fn player_input(keys: Res<Input<KeyCode>>, mut player_query: Query<(&mut CharInput, &mut Player)>) {
    let (mut player_input, mut player) = player_query.single_mut();
    let hold_left = keys.pressed(KeyCode::Left);
    let hold_right = keys.pressed(KeyCode::Right);
    if !(hold_left ^ hold_right) {
        player_input.left = false;
        player_input.right = false;
    } else {
        player_input.left = hold_left;
        player_input.right = hold_right;
    }

    if keys.just_pressed(KeyCode::Left) {
        player.left_pressed = 8;
    }
    if keys.just_pressed(KeyCode::Right) {
        player.right_pressed = 8;
    }
    let can_heavy =
        (player.left_pressed > 0 && hold_left) || (player.right_pressed > 0 && hold_right);
    player.left_pressed = player.left_pressed.saturating_sub(1);
    player.right_pressed = player.right_pressed.saturating_sub(1);

    player_input.jump = keys.just_pressed(KeyCode::Up);
    player_input.shield = keys.pressed(KeyCode::S);
    player_input.grab = keys.just_pressed(KeyCode::Z);
    player_input.light = keys.just_pressed(KeyCode::X) && !can_heavy;
    player_input.heavy = keys.just_pressed(KeyCode::X) && can_heavy;
    player_input.special = keys.just_pressed(KeyCode::C);
}

/// Sent when the game should reset.
pub struct ResetEvent;

/// Resets the game.
fn handle_reset(
    mut commands: Commands,
    mut app_state: ResMut<NextState<AppState>>,
    mut ev_reset: EventReader<ResetEvent>,
    player_query: Query<Entity, With<Player>>,
    bot_query: Query<Entity, With<Bot>>,
) {
    for _ in ev_reset.iter() {
        // Remove player and bot if they exist
        if !player_query.is_empty() {
            let player_e = player_query.single();
            let bot_e = bot_query.single();
            commands.entity(player_e).despawn_recursive();
            commands.entity(bot_e).despawn_recursive();
        }

        // Add player
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
        // Add bot
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

        // We can now run out other systems
        app_state.set(AppState::Running);
    }
}
