use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use bevy_save::AppSaveableExt;
use rand::Rng;

use crate::{
    micro_fighter::{AppState, Floor, SCREEN_SIZE},
    move_states::{CurrentMoveState, MoveState, StateTimer},
};

pub const CHAR_WIDTH: f32 = 20.0;

/// Plugin for general character stuff.
pub struct CharacterPlugin;

impl Plugin for CharacterPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            (
                round_over_on_edge,
                random_actions,
                update_touching_floor,
            )
                .in_set(OnUpdate(AppState::Running)),
        )
        .add_event::<RoundOverEvent>()
        // Serialization
        .register_type::<HorizontalDir>()
        .register_saveable::<Character>()
        .register_saveable::<CharAttrs>()
        .register_saveable::<CharInput>()
        .register_saveable::<Velocity>()
        .register_saveable::<RigidBody>()
        .register_saveable::<GravityScale>();
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Reflect, Default)]
pub enum HorizontalDir {
    #[default]
    Left,
    Right,
}

/// General data for all characters.
#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct Character {
    pub dir: HorizontalDir,
    pub damage: u32,
    pub floor_collider: Option<Entity>,
    pub shielding: bool,
    pub on_floor: bool,
}

/// Components for characters.
#[derive(Bundle)]
pub struct CharBundle {
    pub state_timer: StateTimer,
    pub attrs: CharAttrs,
    pub input: CharInput,
    pub character: Character,
    pub rigidbody: RigidBody,
    pub vel: Velocity,
    pub locked_axes: LockedAxes,
    pub collider: Collider,
    pub transform: TransformBundle,
    pub grav_scale: GravityScale,
    pub sensor: Sensor,
    pub curr_move_state: CurrentMoveState,
}

impl Default for CharBundle {
    fn default() -> Self {
        Self {
            attrs: CharAttrs::default(),
            input: CharInput::default(),
            rigidbody: RigidBody::Dynamic,
            vel: Velocity::default(),
            locked_axes: LockedAxes::ROTATION_LOCKED,
            collider: Collider::cuboid(CHAR_WIDTH / 2.0, 30.0),
            transform: TransformBundle::from(Transform::from_xyz(0.0, 0.0, 0.0)),
            character: Character {
                dir: HorizontalDir::Right,
                floor_collider: None,
                damage: 0,
                shielding: false,
                on_floor: false,
            },
            grav_scale: GravityScale(10.0),
            sensor: Sensor,
            state_timer: StateTimer { frames: 0 },
            curr_move_state: CurrentMoveState {
                move_state: MoveState::Idle,
            },
        }
    }
}

/// Marks a character as controlled by the player.
#[derive(Component)]
pub struct Player {
    pub left_pressed: u8,
    pub right_pressed: u8,
}

#[allow(clippy::derivable_impls)]
impl Default for Player {
    fn default() -> Self {
        Self {
            left_pressed: 0,
            right_pressed: 0,
        }
    }
}

/// Marks a character as controlled by a simple bot.
#[derive(Component)]
pub struct Bot;

/// Holds the character's input for this frame.
#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct CharInput {
    pub shield: bool,
    pub jump: bool,
    pub left: bool,
    pub right: bool,
    pub light: bool,
    pub heavy: bool,
    pub grab: bool,
    pub special: bool,
}

/// Stores character attributes.
#[derive(Component, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct CharAttrs {
    /// Height of a jump, in pixels.
    pub jump_height: u32,
    /// Pixels per second to run at.
    pub run_speed: u32,
    /// Height of the character, in pixels.
    pub size: u32,
    /// Width of the light attack, in pixels.
    pub light_size: u32,
    /// Length of the light attack, in pixels.
    pub light_dist: u32,
    /// Angle of the light attack, in degrees.
    pub light_angle: u32,
    /// Startup time of the light attack, in frames.
    pub light_startup: u32,
    /// Recovery time of the light attack, in frames.
    pub light_recovery: u32,
    /// Light attack's damage amount.
    pub light_dmg: u32,
    /// Width of the heavy attack, in pixels.
    pub heavy_size: u32,
    /// Length of the heavy attack, in pixels.
    pub heavy_dist: u32,
    /// Angle of the heavy attack, in degrees.
    pub heavy_angle: u32,
    /// Startup time of the heavy attack, in frames.
    pub heavy_startup: u32,
    /// Recovery time of the heavy attack, in frames.
    pub heavy_recovery: u32,
    /// Heavy attack's damage amount.
    pub heavy_dmg: u32,
    /// Angle of the throw, in degrees.
    pub throw_angle: u32,
    /// Throwing damage.
    pub throw_dmg: u32,
    /// Speed of the projectile, in pixels per frame.
    pub projectile_speed: u32,
    /// Size of the projectile, in pixels.
    pub projectile_size: u32,
    /// Startup time of spawning a projectile, in frames.
    pub projectile_startup: u32,
    /// Recovery time of spawning a projectile, in frames.
    pub projectile_recovery: u32,
    /// Lifetime of projectile, in frames.
    pub projectile_lifetime: u32,
    /// Projectile damage.
    pub projectile_dmg: u32,
}

impl Default for CharAttrs {
    fn default() -> Self {
        Self {
            jump_height: 150,
            run_speed: 200,
            size: 30,
            light_size: 12,
            light_dist: 12,
            light_angle: 20,
            light_startup: 1,
            light_recovery: 2,
            light_dmg: 4,
            heavy_size: 12,
            heavy_dist: 20,
            heavy_angle: 20,
            heavy_startup: 4,
            heavy_recovery: 8,
            heavy_dmg: 15,
            throw_angle: 45,
            throw_dmg: 6,
            projectile_speed: 8,
            projectile_size: 8,
            projectile_startup: 4,
            projectile_recovery: 2,
            projectile_dmg: 4,
            projectile_lifetime: 30,
        }
    }
}

/// Sent when the game ends.
pub struct RoundOverEvent {
    pub player_won: bool,
}

/// When a character touches one of the screen edges, emit a round over.
fn round_over_on_edge(
    char_query: Query<(&Transform, Option<&Player>), Or<(With<Player>, With<Bot>)>>,
    mut ev_round_over: EventWriter<RoundOverEvent>,
) {
    for (transform, player) in char_query.iter() {
        if transform.translation.x < -((SCREEN_SIZE / 2) as f32)
            || transform.translation.x > (SCREEN_SIZE / 2) as f32
        {
            let player_won = player.is_none();
            ev_round_over.send(RoundOverEvent { player_won });
        }
    }
}

/// Tracks if a character is touching the floor.
fn update_touching_floor(
    mut char_query: Query<(Entity, &mut Character)>,
    mut ev_collision: EventReader<CollisionEvent>,
    floor_query: Query<Entity, With<Floor>>,
) {
    for ev in ev_collision.iter() {
        for (e, mut character) in char_query.iter_mut() {
            let floor_e = floor_query.single();
            let floor_collider = character.floor_collider.unwrap();
            if let CollisionEvent::Started(e1, e2, _) = ev {
                if (*e1 == floor_collider || *e2 == floor_collider)
                    && (*e1 == floor_e || *e2 == floor_e)
                {
                    character.on_floor = true;
                }
            }
            if let CollisionEvent::Stopped(e1, e2, _) = ev {
                if (*e1 == floor_collider || *e2 == floor_collider)
                    && (*e1 == floor_e || *e2 == floor_e)
                {
                    character.on_floor = false;
                }
            }
        }
    }
}

/// Indicates a character should perform random actions.
#[derive(Component)]
pub struct RandomActions;

/// Causes a character to perform random actions.
fn random_actions(mut char_query: Query<&mut CharInput, With<RandomActions>>) {
    for mut char_input in char_query.iter_mut() {
        char_input.left = false;
        char_input.right = false;
        char_input.jump = false;
        char_input.light = false;
        char_input.heavy = false;
        char_input.shield = false;
        char_input.grab = false;

        let mut rng = rand::thread_rng();
        let action = rng.gen_range(0..8);
        match action {
            0 => (),
            1 => char_input.left = true,
            2 => char_input.right = true,
            3 => char_input.jump = true,
            4 => char_input.light = true,
            5 => char_input.heavy = true,
            6 => char_input.shield = true,
            7 => char_input.grab = true,
            _ => unreachable!(),
        }
    }
}