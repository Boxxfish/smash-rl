use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use pyo3::prelude::*;
use rand::Rng;

use crate::character::*;
use crate::hit::*;
use crate::ml::GameState;
use crate::ml::HBox;
use crate::ml::HBoxCollection;
use crate::ml::LoadStateEvent;
use crate::ml::MLBotActionEvent;
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
    /// The state of the game.
    #[pyo3(get)]
    pub game_state: GameState,
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

    /// Runs one step of the environment and returns game info.
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
        let game_state = world.get_resource::<GameState>().unwrap().clone();
        StepOutput {
            hboxes: hbox_coll.hboxes.clone(),
            round_over,
            player_won,
            game_state,
        }
    }

    /// Sends the bot's action this step.
    /// The environment must not be in human mode.
    pub fn bot_step(&mut self, action_id: u32) {
        self.app.world.send_event(MLBotActionEvent { action_id });
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
        let game_state = world.get_resource::<GameState>().unwrap().clone();
        StepOutput {
            hboxes: hbox_coll.hboxes.clone(),
            round_over,
            player_won,
            game_state,
        }
    }

    /// Loads the given state.
    pub fn load_state(&mut self, state: GameState) {
        if self.first_step {
            self.app.setup();
            self.first_step = false;
        }
        self.app
            .world
            .send_event(LoadStateEvent { game_state: state });
        self.app.update();
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
    ev_reset: EventReader<ResetEvent>,
    player_query: Query<Entity, With<Player>>,
    bot_query: Query<Entity, With<Bot>>,
) {
    // This should only run once per frame
    if !ev_reset.is_empty() {
        let (player_e, bot_e) = if !player_query.is_empty() {
            // Remove player and bot children if they exist
            let player_e = player_query.single();
            let bot_e = bot_query.single();
            commands.entity(player_e).clear_children();
            commands.entity(bot_e).clear_children();
            (player_e, bot_e)
        } else {
            // Othewise, create new player and bot
            (
                commands.spawn(Player::default()).id(),
                commands.spawn(Bot).id(),
            )
        };

        // Add player
        let mut rng = rand::thread_rng();
        let p_pos = rng.gen_range(-1.0..1.0) * 50.0;
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
        let mut p_bundle = CharBundle {
            transform: TransformBundle::from(Transform::from_xyz(p_pos, 0.0, 0.0)),
            ..default()
        };
        p_bundle.character.floor_collider = Some(p_floor_collider);
        commands
            .entity(player_e)
            .insert((FallState, p_bundle))
            .add_child(p_floor_collider);

        // Add bot
        let b_pos = rng.gen_range(-1.0..1.0) * 50.0;
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
            transform: TransformBundle::from(Transform::from_xyz(b_pos, 0.0, 0.0)),
            ..default()
        };
        b_bundle.character.floor_collider = Some(b_floor_collider);
        commands
            .entity(bot_e)
            .insert((FallState, b_bundle))
            .add_child(b_floor_collider);

        // We can now run out other systems
        app_state.set(AppState::Running);
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::MicroFighter;

    /// Tests that states can be deterministically restored.
    /// Relies on law of large numbers for exhaustiveness.
    #[test]
    fn restore_state() {
        let mut micro_fighter = MicroFighter::new(false);
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let before_moves: Vec<u32> = (0..100).map(|_| rng.gen_range(0..9)).collect();
            let after_moves: Vec<u32> = (0..100).map(|_| rng.gen_range(0..9)).collect();
            let before_moves_bot: Vec<u32> = (0..100).map(|_| rng.gen_range(0..9)).collect();
            let after_moves_bot: Vec<u32> = (0..100).map(|_| rng.gen_range(0..9)).collect();
            micro_fighter.reset();

            // Run the environment a couple steps
            let mut output = None;
            for (mv, bot_mv) in before_moves.iter().zip(&before_moves_bot) {
                micro_fighter.bot_step(*bot_mv);
                output = Some(micro_fighter.step(*mv));
            }
            let state = output.unwrap().game_state;

            // Collect data after moving a couple more steps
            let mut outputs = Vec::new();
            for (mv, bot_mv) in after_moves.iter().zip(&after_moves_bot) {
                micro_fighter.bot_step(*bot_mv);
                outputs.push(micro_fighter.step(*mv));
            }

            // Reload state and run same steps
            micro_fighter.load_state(state);
            for ((mv, bot_mv), output) in after_moves.iter().zip(&after_moves_bot).zip(&outputs) {
                micro_fighter.bot_step(*bot_mv);
                let new_output = micro_fighter.step(*mv);

                // Check that both outputs match
                assert_eq!(output.round_over, new_output.round_over);
                assert_eq!(output.player_won, new_output.player_won);
                assert_eq!(output.hboxes.len(), new_output.hboxes.len());
                for (h1, h2) in output.hboxes.iter().zip(&new_output.hboxes) {
                    assert_eq!(h1.angle, h2.angle);
                    assert_eq!(h1.move_state, h2.move_state);
                    assert_eq!(h1.damage, h2.damage);
                    assert_eq!(h1.is_hit, h2.is_hit);
                    assert_eq!(h1.is_player, h2.is_player);
                    assert_eq!(h1.x, h2.x);
                    assert_eq!(h1.y, h2.y);
                    assert_eq!(h1.w, h2.w);
                    assert_eq!(h1.h, h2.h);
                }
            }
        }
    }
}
