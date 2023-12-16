use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
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
use crate::ml::NetDamage;
use crate::move_states::*;

pub const FIXED_TIMESTEP: f32 = 1.0 / 30.0;
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
        .add_systems((player_input, reset_on_gameover).in_set(OnUpdate(AppState::Running)));
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
    /// The difference between the damage inflicted by the player and the damage
    /// inflicted on the player.
    #[pyo3(get)]
    pub net_damage: i32,
    /// Samage the player is at.
    #[pyo3(get)]
    pub player_damage: u32,
    /// State of the player.
    #[pyo3(get)]
    pub player_state: MoveState,
    /// Direction the player is facing.
    #[pyo3(get)]
    pub player_dir: i32,
    /// Position of the player.
    #[pyo3(get)]
    pub player_pos: (i32, i32),
    /// Damage the opponent is at.
    #[pyo3(get)]
    pub opponent_damage: u32,
    /// State of the opponent.
    #[pyo3(get)]
    pub opponent_state: MoveState,
    /// Direction the opponent is facing.
    #[pyo3(get)]
    pub opponent_dir: i32,
    /// Position of the opponent.
    #[pyo3(get)]
    pub opponent_pos: (i32, i32),
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
            app.add_plugin(MLPlugin)
                .insert_resource(TimeUpdateStrategy::ManualDuration(
                    std::time::Duration::from_secs_f32(FIXED_TIMESTEP),
                ))
                .insert_resource(RapierConfiguration {
                    timestep_mode: TimestepMode::Fixed {
                        dt: FIXED_TIMESTEP,
                        substeps: 2,
                    },
                    ..default()
                });
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
            self.app.update();
            self.first_step = false;
        }
        self.app.world.send_event(MLPlayerActionEvent { action_id });
        self.app.update();
        self.get_state()
    }

    /// Sends the bot's action this step.
    /// The environment must not be in human mode.
    pub fn bot_step(&mut self, action_id: u32) {
        self.app.world.send_event(MLBotActionEvent { action_id });
    }

    /// Resets the environment.
    pub fn reset(&mut self) -> StepOutput {
        *self = Self::new(false);
        if self.first_step {
            self.app.setup();
            self.app.update();
            self.first_step = false;
        }
        self.get_state()
    }

    /// Loads the given state.
    pub fn load_state(&mut self, state: GameState) -> StepOutput {
        if self.first_step {
            self.app.setup();
            self.app.update();
            self.first_step = false;
        }
        self.app
            .world
            .send_event(LoadStateEvent { game_state: state });
        self.app.update();
        self.get_state()
    }

    /// Returns the current state.
    pub fn get_state(&mut self) -> StepOutput {
        let world = &mut self.app.world;

        let events = world.get_resource::<Events<RoundOverEvent>>().unwrap();
        let mut ev_round_over = events.get_reader();
        let (round_over, player_won) = if let Some(ev) = ev_round_over.iter(events).next() {
            let player_won = ev.player_won;
            (true, player_won)
        } else {
            (false, false)
        };

        let (p_char, p_state, p_xform) = world
            .query_filtered::<(&Character, &CurrentMoveState, &Transform), With<Player>>()
            .single(world);
        let player_damage = p_char.damage;
        let player_state = p_state.move_state;
        let player_dir = match p_char.dir {
            HorizontalDir::Left => -1,
            HorizontalDir::Right => 1,
        };
        let player_pos = (p_xform.translation.x as i32, p_xform.translation.y as i32);

        let (b_char, b_state, b_xform) = world
            .query_filtered::<(&Character, &CurrentMoveState, &Transform), With<Bot>>()
            .single(world);
        let opponent_damage = b_char.damage;
        let opponent_state = b_state.move_state;
        let opponent_dir = match b_char.dir {
            HorizontalDir::Left => -1,
            HorizontalDir::Right => 1,
        };
        let opponent_pos = (b_xform.translation.x as i32, b_xform.translation.y as i32);

        let hbox_coll = world.get_resource::<HBoxCollection>().unwrap();
        let net_dmg = world.get_resource::<NetDamage>().unwrap();
        StepOutput {
            hboxes: hbox_coll.hboxes.clone(),
            round_over,
            player_won,
            net_damage: net_dmg.net_dmg,
            player_damage,
            player_state,
            opponent_damage,
            opponent_state,
            opponent_dir,
            player_dir,
            opponent_pos,
            player_pos,
        }
    }

    /// Returns the current reloadable state of the game.
    pub fn get_game_state(&self) -> GameState {
        self.app.world.get_resource::<GameState>().unwrap().clone()
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
    proj_query: Query<Entity, (With<Hit>, With<Projectile>)>,
) {
    // This should only run once per frame
    if !ev_reset.is_empty() {
        for proj_e in proj_query.iter() {
            commands.get_entity(proj_e).unwrap().despawn();
        }

        if !player_query.is_empty() {
            let player_e = player_query.single();
            let bot_e = bot_query.single();
            commands.entity(player_e).despawn_recursive();
            commands.entity(bot_e).despawn_recursive();
        }
        let player_e = commands.spawn(Player::default()).id();
        let bot_e = commands.spawn(Bot).id();

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

/// Resets the game on game over.
/// Only used on the human version.
fn reset_on_gameover(
    mut ev_reset: EventWriter<ResetEvent>,
    mut ev_round_over: EventReader<RoundOverEvent>,
) {
    for _ in ev_round_over.iter() {
        ev_reset.send(ResetEvent);
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
        for i in 0..50 {
            let before_moves: Vec<u32> = (0..100).map(|_| rng.gen_range(0..9)).collect();
            let after_moves: Vec<u32> = (0..100).map(|_| rng.gen_range(0..9)).collect();
            let before_moves_bot: Vec<u32> = (0..100).map(|_| rng.gen_range(0..9)).collect();
            let after_moves_bot: Vec<u32> = (0..100).map(|_| rng.gen_range(0..9)).collect();
            micro_fighter.reset();

            // Run the environment a couple steps
            for (mv, bot_mv) in before_moves.iter().zip(&before_moves_bot) {
                micro_fighter.bot_step(*bot_mv);
                micro_fighter.step(*mv);
            }
            let state = micro_fighter.get_game_state();

            // Collect data after moving a couple more steps
            let mut outputs = Vec::new();
            for (mv, bot_mv) in after_moves.iter().zip(&after_moves_bot) {
                micro_fighter.bot_step(*bot_mv);
                outputs.push(micro_fighter.step(*mv));
            }

            // Reload state and run same steps
            micro_fighter.load_state(state);
            for (j, ((mv, bot_mv), output)) in after_moves
                .iter()
                .zip(&after_moves_bot)
                .zip(&outputs)
                .enumerate()
            {
                println!("Testing round {i}, move {j}. Move was {mv}");
                micro_fighter.bot_step(*bot_mv);
                let new_output = micro_fighter.step(*mv);

                // Check that both outputs match
                assert_eq!(output.round_over, new_output.round_over);
                assert_eq!(output.player_won, new_output.player_won);
                assert_eq!(output.hboxes.len(), new_output.hboxes.len());
                for (h1, h2) in output.hboxes.iter().zip(&new_output.hboxes) {
                    assert_eq!(h1.angle, h2.angle);
                    assert_eq!(h1.damage, h2.damage);
                    assert_eq!(h1.is_hit, h2.is_hit);
                    assert_eq!(h1.is_player, h2.is_player);
                    assert!(h1.x.abs_diff(h2.x) <= 4);
                    assert!(h1.y.abs_diff(h2.y) <= 4);
                    assert_eq!(h1.w, h2.w);
                    assert_eq!(h1.h, h2.h);
                }
            }
        }
    }
}
