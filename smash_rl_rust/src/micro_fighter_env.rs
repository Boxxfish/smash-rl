use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use pyo3::prelude::*;

const FIXED_TIMESTEP: f32 = 1.0 / 60.0;
const JUMP_VEL: f32 = 500.0;
const AIR_VEL: f32 = 2.0;
const CHAR_WIDTH: f32 = 20.0;
const HIT_OFFSET: Vec2 = Vec2 {
    x: CHAR_WIDTH / 2.0,
    y: 0.0,
};

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
            .add_state::<AppState>()
            .add_startup_system(setup)
            .add_systems(
                (
                    player_input,
                    handle_idle,
                    handle_run,
                    handle_jump,
                    handle_fall,
                    handle_light_attack_startup,
                    handle_light_attack_hit,
                    handle_light_attack_recovery,
                )
                    .in_set(OnUpdate(AppState::Running)), // .in_schedule(CoreSchedule::FixedUpdate)
            )
            // .insert_resource(FixedTime::new_from_secs(FIXED_TIMESTEP))
            .run();
    }
}

impl Default for MicroFighterEnv {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum HorizontalDir {
    Left,
    Right,
}

/// General data for all characters.
#[derive(Component)]
pub struct Character {
    pub dir: HorizontalDir,
    pub floor_collider: Option<Entity>,
}

/// Components for characters.
#[derive(Bundle)]
pub struct CharBundle {
    pub state: IdleState,
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
}

impl Default for CharBundle {
    fn default() -> Self {
        Self {
            state: IdleState,
            attrs: CharAttrs::default(),
            input: CharInput::default(),
            rigidbody: RigidBody::Dynamic,
            vel: Velocity::default(),
            locked_axes: LockedAxes::ROTATION_LOCKED,
            collider: Collider::cuboid(CHAR_WIDTH / 2.0, 30.0),
            transform: TransformBundle::from(Transform::from_xyz(-50.0, 0.0, 0.0)),
            character: Character {
                dir: HorizontalDir::Right,
                floor_collider: None,
            },
            grav_scale: GravityScale(10.0),
            sensor: Sensor,
        }
    }
}

/// Denotes that the entity is a hitbox.
#[derive(Component)]
pub struct Hit;

/// Bundle for hits.
#[derive(Bundle)]
pub struct HitBundle {
    pub hit: Hit,
    pub collider: Collider,
    pub transform: TransformBundle,
    pub active_events: ActiveEvents,
}

impl HitBundle {
    fn new(size: u32, dist: u32, angle: u32, offset: Vec2, dir: HorizontalDir) -> Self {
        let dir_mult = match dir {
            HorizontalDir::Left => -1.0,
            HorizontalDir::Right => 1.0,
        };
        let mut translation = Vec2::new(
            (angle as f32).to_radians().cos() * dist as f32 / 2.0,
            (angle as f32).to_radians().sin() * dist as f32 / 2.0 - (angle as f32).to_radians().cos() * size as f32 / 2.0,
        ) + offset;
        translation.x *= dir_mult;
        Self {
            hit: Hit,
            collider: Collider::cuboid(dist as f32 / 2.0, size as f32 / 2.0),
            transform: TransformBundle::from(Transform {
                translation: translation.extend(0.0),
                rotation: Quat::from_rotation_z((angle as f32 * dir_mult).to_radians()),
                ..default()
            }),
            active_events: ActiveEvents::COLLISION_EVENTS,
        }
    }
}

fn setup(mut commands: Commands, mut app_state: ResMut<NextState<AppState>>) {
    commands.spawn(Camera2dBundle::default());
    // Floor
    commands.spawn((
        Collider::cuboid(500.0, 50.0),
        TransformBundle::from(Transform::from_xyz(0.0, -200.0, 0.0)),
    ));
    // Player
    let p_floor_collider = commands
        .spawn((
            Collider::cuboid(8.0, 4.0),
            TransformBundle::from(Transform::from_xyz(0.0, -30.0, 0.0)),
            ActiveEvents::COLLISION_EVENTS,
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
#[derive(Component, Default)]
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
#[derive(Component)]
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
    /// Speed of the projectile, in pixels per second.
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
            jump_height: 40,
            run_speed: 200,
            size: 30,
            light_size: 12,
            light_dist: 12,
            light_angle: 20,
            light_startup: 0,
            light_recovery: 2,
            light_dmg: 2,
            heavy_size: 12,
            heavy_dist: 20,
            heavy_angle: 20,
            heavy_startup: 4,
            heavy_recovery: 8,
            heavy_dmg: 12,
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
        player.left_pressed = 4;
    }
    if keys.just_pressed(KeyCode::Right) {
        player.right_pressed = 4;
    }
    let can_heavy =
        (player.left_pressed > 0 && hold_left) || (player.right_pressed > 0 && hold_right);
    player.left_pressed = player.left_pressed.saturating_sub(1);
    player.right_pressed = player.right_pressed.saturating_sub(1);

    player_input.jump = keys.just_pressed(KeyCode::Up);
    player_input.shield = keys.just_pressed(KeyCode::S);
    player_input.grab = keys.just_pressed(KeyCode::Z);
    player_input.light = keys.just_pressed(KeyCode::X) && !can_heavy;
    player_input.heavy = keys.just_pressed(KeyCode::X) && can_heavy;
    player_input.special = keys.just_pressed(KeyCode::C);
}

#[derive(Component)]
pub struct IdleState;
#[derive(Component)]
pub struct RunState;

#[derive(Component)]
pub struct JumpState {
    pub frames_left: u32,
}
#[derive(Component)]
pub struct FallState;
#[derive(Component)]
pub struct ShieldState;
#[derive(Component)]
pub struct HitstunState;
#[derive(Component)]
pub struct LightAttackStartupState {
    pub frames_left: u32,
}
#[derive(Component)]
pub struct LightAttackHitState {
    pub frames_left: u32,
}
#[derive(Component)]
pub struct LightAttackRecoveryState {
    pub frames_left: u32,
}
#[derive(Component)]
pub struct HeavyAttackState;
#[derive(Component)]
pub struct GrabState;

fn handle_idle(
    mut char_query: Query<(Entity, &CharInput, &CharAttrs, &mut Character), With<IdleState>>,
    mut commands: Commands,
) {
    for (e, char_inpt, char_attrs, mut character) in char_query.iter_mut() {
        if char_inpt.jump {
            commands
                .entity(e)
                .insert(JumpState {
                    frames_left: ((char_attrs.jump_height as f32 / JUMP_VEL) / FIXED_TIMESTEP)
                        as u32,
                })
                .remove::<IdleState>();
        } else if char_inpt.left {
            character.dir = HorizontalDir::Left;
            commands.entity(e).insert(RunState).remove::<IdleState>();
        } else if char_inpt.right {
            character.dir = HorizontalDir::Right;
            commands.entity(e).insert(RunState).remove::<IdleState>();
        } else if char_inpt.light {
            commands
                .entity(e)
                .insert(LightAttackStartupState {
                    frames_left: char_attrs.light_startup,
                })
                .remove::<IdleState>();
        }
    }
}

fn handle_run(
    mut char_query: Query<
        (Entity, &CharInput, &CharAttrs, &mut Velocity, &Character),
        With<RunState>,
    >,
    mut commands: Commands,
) {
    for (e, char_inpt, char_attrs, mut vel, character) in char_query.iter_mut() {
        let dir_mult = if character.dir == HorizontalDir::Left {
            -1.0
        } else {
            1.0
        };
        vel.linvel = Vec2::new(char_attrs.run_speed as f32 * dir_mult, 0.0);
        if char_inpt.jump {
            commands
                .entity(e)
                .insert((
                    JumpState {
                        frames_left: ((char_attrs.jump_height as f32 / JUMP_VEL) / FIXED_TIMESTEP)
                            as u32,
                    },
                    GravityScale(0.0),
                ))
                .remove::<RunState>();
            if char_inpt.left {
                vel.linvel.x = -(char_attrs.run_speed as f32) / 2.0;
            } else if char_inpt.right {
                vel.linvel.x = char_attrs.run_speed as f32 / 2.0;
            }
        } else if (!char_inpt.left && character.dir == HorizontalDir::Left)
            || (!char_inpt.right && character.dir == HorizontalDir::Right)
        {
            vel.linvel = Vec2::new(0.0, 0.0);
            commands.entity(e).insert(IdleState).remove::<RunState>();
        }
    }
}

fn handle_jump(
    mut char_query: Query<(
        Entity,
        &CharInput,
        &CharAttrs,
        &mut Velocity,
        &mut JumpState,
    )>,
    mut commands: Commands,
) {
    for (e, char_inpt, char_attrs, mut vel, mut jump_state) in char_query.iter_mut() {
        vel.linvel.y = JUMP_VEL;
        jump_state.frames_left -= 1;
        if jump_state.frames_left == 0 {
            commands
                .entity(e)
                .insert((FallState, GravityScale(10.0)))
                .remove::<JumpState>();
        }
        if char_inpt.left {
            vel.linvel.x += -AIR_VEL;
        } else if char_inpt.right {
            vel.linvel.x += AIR_VEL;
        }
    }
}

fn handle_fall(
    mut char_query: Query<(Entity, &CharInput, &mut Velocity, &Character, &FallState)>,
    mut commands: Commands,
    mut ev_collision: EventReader<CollisionEvent>,
) {
    for (e, char_inpt, mut vel, character, _) in char_query.iter_mut() {
        for ev in ev_collision.iter() {
            if let CollisionEvent::Started(e1, e2, _) = ev {
                let floor_e = character.floor_collider.unwrap();
                if *e1 == floor_e || *e2 == floor_e {
                    commands.entity(e).insert(IdleState).remove::<FallState>();
                }
            }
        }
        if char_inpt.left {
            vel.linvel.x += -AIR_VEL;
        } else if char_inpt.right {
            vel.linvel.x += AIR_VEL;
        }
    }
}

fn handle_light_attack_startup(
    mut char_query: Query<(Entity, &CharAttrs, &Character, &mut LightAttackStartupState)>,
    mut commands: Commands,
) {
    for (e, char_attrs, character, mut startup_state) in char_query.iter_mut() {
        if startup_state.frames_left == 0 {
            let hit = commands
                .spawn(HitBundle::new(
                    char_attrs.light_size,
                    char_attrs.light_dist,
                    char_attrs.light_angle,
                    HIT_OFFSET,
                    character.dir,
                ))
                .id();
            commands.entity(e).add_child(hit);

            commands
                .entity(e)
                .insert(LightAttackHitState { frames_left: 2 })
                .remove::<LightAttackStartupState>();
        }
        startup_state.frames_left = startup_state.frames_left.saturating_sub(1);
    }
}

fn handle_light_attack_hit(
    mut char_query: Query<(Entity, &CharAttrs, &mut LightAttackHitState)>,
    hit_query: Query<(Entity, &Parent), With<Hit>>,
    mut commands: Commands,
) {
    for (e, char_attrs, mut hit_state) in char_query.iter_mut() {
        if hit_state.frames_left == 0 {
            for (hit_e, hit_parent) in hit_query.iter() {
                if hit_parent.get() == e {
                    commands.entity(hit_e).despawn();
                }
            }

            commands
                .entity(e)
                .insert(LightAttackRecoveryState {
                    frames_left: char_attrs.light_recovery,
                })
                .remove::<LightAttackHitState>();
        }
        hit_state.frames_left = hit_state.frames_left.saturating_sub(1);
    }
}

fn handle_light_attack_recovery(
    mut char_query: Query<(Entity, &mut LightAttackRecoveryState)>,
    mut commands: Commands,
) {
    for (e, mut recovery_state) in char_query.iter_mut() {
        if recovery_state.frames_left == 0 {
            commands
                .entity(e)
                .insert(IdleState)
                .remove::<LightAttackRecoveryState>();
        }
        recovery_state.frames_left = recovery_state.frames_left.saturating_sub(1);
    }
}
