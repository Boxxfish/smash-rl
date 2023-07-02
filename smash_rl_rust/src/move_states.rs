use std::marker::PhantomData;

use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use pyo3::prelude::*;

use crate::{
    character::{CharAttrs, CharInput, Character, HorizontalDir, CHAR_WIDTH},
    hit::{Hit, HitBundle, HitType, Projectile, Hitstun},
    micro_fighter::{AppState, Floor, FIXED_TIMESTEP},
};

const JUMP_VEL: f32 = 500.0;
const AIR_VEL: f32 = 2.0;
const HIT_OFFSET: Vec2 = Vec2 {
    x: CHAR_WIDTH / 2.0,
    y: 0.0,
};
const GRAB_RECOVERY: u32 = 10;
const GRAB_SIZE: u32 = 4;

/// Plugin for move states.
pub struct MoveStatesPlugin;

impl Plugin for MoveStatesPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            (
                handle_idle,
                handle_run,
                handle_jump,
                handle_fall,
                handle_light_attack_startup,
                handle_light_attack_hit_start,
                handle_light_attack_hit,
                handle_light_attack_hit_end,
                handle_light_attack_recovery,
                handle_heavy_attack_startup,
                handle_heavy_attack_hit_start,
                handle_heavy_attack_hit,
                handle_heavy_attack_hit_end,
                handle_heavy_attack_recovery,
            )
                .in_set(OnUpdate(AppState::Running)),
        )
        .add_systems(
            (
                handle_special_attack_startup,
                handle_special_attack_hit,
                handle_special_attack_recovery,
            )
                .in_set(OnUpdate(AppState::Running)),
        )
        .add_systems(
            (
                handle_shield_start,
                handle_shield,
                handle_shield_end,
                handle_grab_start,
                handle_grab,
                handle_grab_end,
                handle_hitstun,
                update_timer,
            )
                .in_set(OnUpdate(AppState::Running)),
        )
        .add_plugin(MoveStatePlugin::<IdleState, { MoveState::Idle as u32 }>::default())
        .add_plugin(MoveStatePlugin::<RunState, { MoveState::Run as u32 }>::default())
        .add_plugin(MoveStatePlugin::<JumpState, { MoveState::Jump as u32 }>::default())
        .add_plugin(MoveStatePlugin::<FallState, { MoveState::Fall as u32 }>::default())
        .add_plugin(MoveStatePlugin::<
            LightAttackStartupState,
            { MoveState::LightAttackStartup as u32 },
        >::default())
        .add_plugin(MoveStatePlugin::<
            LightAttackHitState,
            { MoveState::LightAttackHit as u32 },
        >::default())
        .add_plugin(MoveStatePlugin::<
            LightAttackRecoveryState,
            { MoveState::LightAttackRecovery as u32 },
        >::default())
        .add_plugin(MoveStatePlugin::<
            HeavyAttackStartupState,
            { MoveState::HeavyAttackStartup as u32 },
        >::default())
        .add_plugin(MoveStatePlugin::<
            HeavyAttackHitState,
            { MoveState::HeavyAttackHit as u32 },
        >::default())
        .add_plugin(MoveStatePlugin::<
            HeavyAttackRecoveryState,
            { MoveState::HeavyAttackRecovery as u32 },
        >::default())
        .add_plugin(MoveStatePlugin::<
            SpecialAttackStartupState,
            { MoveState::SpecialAttackStartup as u32 },
        >::default())
        .add_plugin(MoveStatePlugin::<
            SpecialAttackHitState,
            { MoveState::SpecialAttackHit as u32 },
        >::default())
        .add_plugin(MoveStatePlugin::<
            SpecialAttackRecoveryState,
            { MoveState::SpecialAttackRecovery as u32 },
        >::default())
        .add_plugin(MoveStatePlugin::<GrabState, { MoveState::Grab as u32 }>::default())
        .add_plugin(MoveStatePlugin::<ShieldState, { MoveState::Shield as u32 }>::default())
        .add_plugin(MoveStatePlugin::<HitstunState, { MoveState::Hitstun as u32 }>::default());
    }
}

/// Tracks the current move state.
#[derive(Component)]
pub struct CurrentMoveState {
    pub move_state: MoveState,
}

/// Enumerates states a character can be in.
#[pyclass]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoveState {
    Idle = 0,
    Run = 1,
    Jump = 2,
    Fall = 3,
    Shield = 4,
    Hitstun = 5,
    LightAttackStartup = 6,
    LightAttackHit = 7,
    LightAttackRecovery = 8,
    HeavyAttackStartup = 9,
    HeavyAttackHit = 10,
    HeavyAttackRecovery = 11,
    SpecialAttackStartup = 12,
    SpecialAttackHit = 13,
    SpecialAttackRecovery = 14,
    Grab = 15,
}

impl From<u32> for MoveState {
    fn from(value: u32) -> Self {
        match value {
            0 => MoveState::Idle,
            1 => MoveState::Run,
            2 => MoveState::Jump,
            3 => MoveState::Fall,
            4 => MoveState::Shield,
            5 => MoveState::Hitstun,
            6 => MoveState::LightAttackStartup,
            7 => MoveState::LightAttackHit,
            8 => MoveState::LightAttackRecovery,
            9 => MoveState::HeavyAttackStartup,
            10 => MoveState::HeavyAttackHit,
            11 => MoveState::HeavyAttackRecovery,
            12 => MoveState::SpecialAttackStartup,
            13 => MoveState::SpecialAttackHit,
            14 => MoveState::SpecialAttackRecovery,
            15 => MoveState::Grab,
            _ => unimplemented!("Cannot convert this number to MoveState."),
        }
    }
}

/// Given a MoveState, adds the appropriate state to an entity.
pub fn add_move_state(move_state: MoveState, entity: Entity, commands: &mut Commands) {
    let mut e_cmds = commands.get_entity(entity).unwrap();
    match move_state {
        MoveState::Idle => e_cmds.insert(IdleState),
        MoveState::Run => e_cmds.insert(RunState),
        MoveState::Jump => e_cmds.insert(JumpState),
        MoveState::Fall => e_cmds.insert(FallState),
        MoveState::Shield => e_cmds.insert(ShieldState),
        MoveState::Hitstun => e_cmds.insert(HitstunState),
        MoveState::LightAttackStartup => e_cmds.insert(LightAttackStartupState),
        MoveState::LightAttackHit => e_cmds.insert(LightAttackHitState),
        MoveState::LightAttackRecovery => e_cmds.insert(LightAttackRecoveryState),
        MoveState::HeavyAttackStartup => e_cmds.insert(HeavyAttackStartupState),
        MoveState::HeavyAttackHit => e_cmds.insert(HeavyAttackHitState),
        MoveState::HeavyAttackRecovery => e_cmds.insert(HeavyAttackRecoveryState),
        MoveState::SpecialAttackStartup => e_cmds.insert(SpecialAttackStartupState),
        MoveState::SpecialAttackHit => e_cmds.insert(SpecialAttackHitState),
        MoveState::SpecialAttackRecovery => e_cmds.insert(SpecialAttackRecoveryState),
        MoveState::Grab => e_cmds.insert(GrabState),
    };
}

#[derive(Component)]
pub struct IdleState;
#[derive(Component)]
pub struct RunState;
#[derive(Component)]
pub struct JumpState;
#[derive(Component)]
pub struct FallState;
#[derive(Component)]
pub struct ShieldState;
#[derive(Component)]
pub struct HitstunState;
#[derive(Component)]
pub struct LightAttackStartupState;
#[derive(Component)]
pub struct LightAttackHitState;
#[derive(Component)]
pub struct LightAttackRecoveryState;
#[derive(Component)]
pub struct HeavyAttackStartupState;
#[derive(Component)]
pub struct HeavyAttackHitState;
#[derive(Component)]
pub struct HeavyAttackRecoveryState;
#[derive(Component)]
pub struct SpecialAttackStartupState;
#[derive(Component)]
pub struct SpecialAttackHitState;
#[derive(Component)]
pub struct SpecialAttackRecoveryState;
#[derive(Component)]
pub struct GrabState;

/// Holds the current frame since this state started.
#[derive(Component)]
pub struct StateTimer {
    pub frames: u32,
}

/// Treats the component as a move state.
struct MoveStatePlugin<T: Component, const U: u32> {
    t: PhantomData<T>,
}

impl<T: Component, const U: u32> Default for MoveStatePlugin<T, U> {
    fn default() -> Self {
        Self {
            t: Default::default(),
        }
    }
}

impl<T: Component, const U: u32> Plugin for MoveStatePlugin<T, U> {
    fn build(&self, app: &mut App) {
        app.add_systems(
            (
                reset_state_timer::<T>,
                exit_on_hitstun::<T>,
                update_move_state::<T, U>,
            )
                .in_set(OnUpdate(AppState::Running)),
        );
    }
}

/// Resets the state timer whenever the component is added.
fn reset_state_timer<T: Component>(mut timer_query: Query<&mut StateTimer, Added<T>>) {
    for mut timer in timer_query.iter_mut() {
        timer.frames = 0;
    }
}

/// Updates the current state whenever the component is added.
fn update_move_state<T: Component, const U: u32>(
    mut ms_query: Query<&mut CurrentMoveState, Added<T>>,
) {
    for mut curr_move_state in ms_query.iter_mut() {
        curr_move_state.move_state = MoveState::from(U);
    }
}

fn exit_on_hitstun<T: Component>(
    char_query: Query<(Entity, Option<&T>), Added<HitstunState>>,
    mut commands: Commands,
) {
    for (e, state) in char_query.iter() {
        if state.is_some() {
            commands.entity(e).remove::<T>();
        }
    }
}

/// Increases the frame count every frame.
fn update_timer(mut timer_query: Query<&mut StateTimer>) {
    for mut timer in timer_query.iter_mut() {
        timer.frames += 1;
    }
}

fn handle_idle(
    mut char_query: Query<(Entity, &CharInput, &mut Character), With<IdleState>>,
    mut commands: Commands,
) {
    for (e, char_inpt, mut character) in char_query.iter_mut() {
        if char_inpt.jump {
            commands.entity(e).insert(JumpState).remove::<IdleState>();
        } else if char_inpt.left {
            character.dir = HorizontalDir::Left;
            commands.entity(e).insert(RunState).remove::<IdleState>();
        } else if char_inpt.right {
            character.dir = HorizontalDir::Right;
            commands.entity(e).insert(RunState).remove::<IdleState>();
        } else if char_inpt.light {
            commands
                .entity(e)
                .insert(LightAttackStartupState)
                .remove::<IdleState>();
        } else if char_inpt.heavy {
            commands
                .entity(e)
                .insert(HeavyAttackStartupState)
                .remove::<IdleState>();
        } else if char_inpt.special {
            commands
                .entity(e)
                .insert(SpecialAttackStartupState)
                .remove::<IdleState>();
        } else if char_inpt.shield {
            commands.entity(e).insert(ShieldState).remove::<IdleState>();
        } else if char_inpt.grab {
            commands.entity(e).insert(GrabState).remove::<IdleState>();
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
                .insert((JumpState, GravityScale(0.0)))
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
        } else if char_inpt.heavy {
            vel.linvel = Vec2::new(0.0, 0.0);
            commands
                .entity(e)
                .insert(HeavyAttackStartupState)
                .remove::<RunState>();
        }
    }
}

fn handle_jump(
    mut char_query: Query<
        (Entity, &CharInput, &CharAttrs, &mut Velocity, &StateTimer),
        With<JumpState>,
    >,
    mut commands: Commands,
) {
    for (e, char_inpt, char_attrs, mut vel, timer) in char_query.iter_mut() {
        vel.linvel.y = JUMP_VEL;
        if timer.frames >= ((char_attrs.jump_height as f32 / JUMP_VEL) / FIXED_TIMESTEP) as u32 {
            vel.linvel.y = 0.0;
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
    mut char_query: Query<(Entity, &CharInput, &mut Velocity, &Character), With<FallState>>,
    floor_query: Query<Entity, With<Floor>>,
    mut commands: Commands,
    mut ev_collision: EventReader<CollisionEvent>,
) {
    // Go to idle if touching floor
    for ev in ev_collision.iter() {
        for (e, _, _, character) in char_query.iter() {
            let floor_e = floor_query.single();
            if let CollisionEvent::Started(e1, e2, _) = ev {
                let floor_collider = character.floor_collider.unwrap();
                if (*e1 == floor_collider || *e2 == floor_collider)
                    && (*e1 == floor_e || *e2 == floor_e)
                {
                    commands.entity(e).insert(IdleState).remove::<FallState>();
                }
            }
        }
    }

    // Handle moving horizontally
    for (_, char_inpt, mut vel, _) in char_query.iter_mut() {
        if char_inpt.left {
            vel.linvel.x += -AIR_VEL;
        } else if char_inpt.right {
            vel.linvel.x += AIR_VEL;
        }
    }
}

fn handle_light_attack_startup(
    char_query: Query<(Entity, &CharAttrs, &StateTimer), With<LightAttackStartupState>>,
    mut commands: Commands,
) {
    for (e, char_attrs, timer) in char_query.iter() {
        if timer.frames == char_attrs.light_startup {
            commands
                .entity(e)
                .insert(LightAttackHitState)
                .remove::<LightAttackStartupState>();
        }
    }
}

fn handle_light_attack_hit_start(
    char_query: Query<(Entity, &CharAttrs, &Character), Added<LightAttackHitState>>,
    mut commands: Commands,
) {
    for (e, char_attrs, character) in char_query.iter() {
        let hit = commands
            .spawn(HitBundle::new(
                char_attrs.light_dmg,
                char_attrs.light_size,
                char_attrs.light_dist,
                char_attrs.light_angle,
                HIT_OFFSET,
                character.dir,
                e,
                HitType::Normal,
            ))
            .id();
        commands.entity(e).add_child(hit);
    }
}

fn handle_light_attack_hit(
    char_query: Query<(Entity, &StateTimer), With<LightAttackHitState>>,
    mut commands: Commands,
) {
    for (e, timer) in char_query.iter() {
        if timer.frames == 4 {
            commands
                .entity(e)
                .insert(LightAttackRecoveryState)
                .remove::<LightAttackHitState>();
        }
    }
}

fn handle_light_attack_hit_end(
    mut rem_query: RemovedComponents<LightAttackHitState>,
    hit_query: Query<(Entity, &Parent), With<Hit>>,
    mut commands: Commands,
) {
    for e in rem_query.iter() {
        for (hit_e, hit_parent) in hit_query.iter() {
            if hit_parent.get() == e {
                commands.entity(hit_e).remove_parent().despawn();
            }
        }
    }
}

fn handle_light_attack_recovery(
    char_query: Query<(Entity, &CharAttrs, &StateTimer), With<LightAttackRecoveryState>>,
    mut commands: Commands,
) {
    for (e, char_attrs, timer) in char_query.iter() {
        if timer.frames == char_attrs.light_recovery {
            commands
                .entity(e)
                .insert(IdleState)
                .remove::<LightAttackRecoveryState>();
        }
    }
}

fn handle_heavy_attack_startup(
    char_query: Query<(Entity, &CharAttrs, &StateTimer), With<HeavyAttackStartupState>>,
    mut commands: Commands,
) {
    for (e, char_attrs, timer) in char_query.iter() {
        if timer.frames == char_attrs.heavy_startup {
            commands
                .entity(e)
                .insert(HeavyAttackHitState)
                .remove::<HeavyAttackStartupState>();
        }
    }
}

fn handle_heavy_attack_hit_start(
    char_query: Query<(Entity, &CharAttrs, &Character), Added<HeavyAttackHitState>>,
    mut commands: Commands,
) {
    for (e, char_attrs, character) in char_query.iter() {
        let hit = commands
            .spawn(HitBundle::new(
                char_attrs.heavy_dmg,
                char_attrs.heavy_size,
                char_attrs.heavy_dist,
                char_attrs.heavy_angle,
                HIT_OFFSET,
                character.dir,
                e,
                HitType::Normal,
            ))
            .id();
        commands.entity(e).add_child(hit);
    }
}

fn handle_heavy_attack_hit(
    char_query: Query<(Entity, &StateTimer), With<HeavyAttackHitState>>,
    mut commands: Commands,
) {
    for (e, timer) in char_query.iter() {
        if timer.frames == 4 {
            commands
                .entity(e)
                .insert(HeavyAttackRecoveryState)
                .remove::<HeavyAttackHitState>();
        }
    }
}

fn handle_heavy_attack_hit_end(
    mut rem_query: RemovedComponents<HeavyAttackHitState>,
    hit_query: Query<(Entity, &Parent), With<Hit>>,
    mut commands: Commands,
) {
    for e in rem_query.iter() {
        for (hit_e, hit_parent) in hit_query.iter() {
            if hit_parent.get() == e {
                commands.entity(hit_e).remove_parent().despawn();
            }
        }
    }
}

fn handle_heavy_attack_recovery(
    char_query: Query<(Entity, &CharAttrs, &StateTimer), With<HeavyAttackRecoveryState>>,
    mut commands: Commands,
) {
    for (e, char_attrs, timer) in char_query.iter() {
        if timer.frames == char_attrs.heavy_recovery {
            commands
                .entity(e)
                .insert(IdleState)
                .remove::<HeavyAttackRecoveryState>();
        }
    }
}

fn handle_shield_start(mut char_query: Query<&mut Character, Added<ShieldState>>) {
    for mut character in char_query.iter_mut() {
        character.shielding = true;
    }
}

fn handle_shield(
    char_query: Query<(Entity, &CharInput), With<ShieldState>>,
    mut commands: Commands,
) {
    for (e, char_inpt) in char_query.iter() {
        if !char_inpt.shield {
            commands.entity(e).insert(IdleState).remove::<ShieldState>();
        }
    }
}

fn handle_shield_end(
    mut rem_query: RemovedComponents<ShieldState>,
    mut char_query: Query<&mut Character>,
) {
    for e in rem_query.iter() {
        if let Ok(mut character) = char_query.get_mut(e) {
            character.shielding = false;
        }
    }
}

fn handle_special_attack_startup(
    char_query: Query<(Entity, &CharAttrs, &StateTimer), With<SpecialAttackStartupState>>,
    mut commands: Commands,
) {
    for (e, char_attrs, timer) in char_query.iter() {
        if timer.frames == char_attrs.projectile_startup {
            commands
                .entity(e)
                .insert(SpecialAttackHitState)
                .remove::<SpecialAttackStartupState>();
        }
    }
}

fn handle_special_attack_hit(
    char_query: Query<(Entity, &CharAttrs, &Character, &Transform), With<SpecialAttackHitState>>,
    mut commands: Commands,
) {
    for (e, char_attrs, character, transform) in char_query.iter() {
        let mut hit_bundle = HitBundle::new(
            char_attrs.projectile_dmg,
            char_attrs.projectile_size,
            char_attrs.projectile_size,
            0,
            Vec2::ZERO,
            character.dir,
            e,
            HitType::Normal,
        );
        hit_bundle.transform.local.translation = transform.translation;
        commands.spawn((
            Projectile {
                frames_left: char_attrs.projectile_lifetime,
                speed: char_attrs.projectile_speed,
                dir: character.dir,
            },
            hit_bundle,
        ));

        commands
            .entity(e)
            .insert(SpecialAttackRecoveryState)
            .remove::<SpecialAttackHitState>();
    }
}

fn handle_special_attack_recovery(
    char_query: Query<(Entity, &CharAttrs, &StateTimer), With<SpecialAttackRecoveryState>>,
    mut commands: Commands,
) {
    for (e, char_attrs, timer) in char_query.iter() {
        if timer.frames == char_attrs.heavy_recovery {
            commands
                .entity(e)
                .insert(IdleState)
                .remove::<SpecialAttackRecoveryState>();
        }
    }
}

fn handle_grab_start(
    char_query: Query<(Entity, &CharAttrs, &Character), Added<GrabState>>,
    mut commands: Commands,
) {
    for (e, char_attrs, character) in char_query.iter() {
        let hit = commands
            .spawn(HitBundle::new(
                char_attrs.throw_dmg,
                GRAB_SIZE,
                GRAB_SIZE,
                char_attrs.throw_angle,
                HIT_OFFSET,
                character.dir,
                e,
                HitType::Grab,
            ))
            .id();
        commands.entity(e).add_child(hit);
    }
}

fn handle_grab(char_query: Query<(Entity, &StateTimer), With<GrabState>>, mut commands: Commands) {
    for (e, timer) in char_query.iter() {
        if timer.frames == GRAB_RECOVERY {
            commands.entity(e).insert(IdleState).remove::<GrabState>();
        }
    }
}

fn handle_grab_end(
    mut rem_query: RemovedComponents<GrabState>,
    hit_query: Query<(Entity, &Parent), With<Hit>>,
    mut commands: Commands,
) {
    for e in rem_query.iter() {
        for (hit_e, hit_parent) in hit_query.iter() {
            if hit_parent.get() == e {
                commands.entity(hit_e).remove_parent().despawn();
            }
        }
    }
}

fn handle_hitstun(
    mut char_query: Query<(Entity, &Character, &HitstunState, &Hitstun, &StateTimer)>,
    floor_query: Query<Entity, With<Floor>>,
    mut commands: Commands,
    mut ev_collision: EventReader<CollisionEvent>,
) {
    for (e, character, hitstun_state, hitstun, timer) in char_query.iter_mut() {
        if timer.frames == hitstun.frames {
            // Check if on ground
            let mut on_ground = false;
            let floor_e = floor_query.single();
            for ev in ev_collision.iter() {
                if let CollisionEvent::Started(e1, e2, _) = ev {
                    let floor_collider = character.floor_collider.unwrap();
                    if (*e1 == floor_collider || *e2 == floor_collider)
                        && (*e1 == floor_e || *e2 == floor_e)
                    {
                        on_ground = true;
                        break;
                    }
                }
            }

            if on_ground {
                commands
                    .entity(e)
                    .insert(IdleState)
                    .remove::<HitstunState>()
                    .remove::<Hitstun>();
            } else {
                commands
                    .entity(e)
                    .insert(FallState)
                    .remove::<HitstunState>()
                    .remove::<Hitstun>();
            }
        }
    }
}
