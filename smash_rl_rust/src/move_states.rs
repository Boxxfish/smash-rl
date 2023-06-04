use bevy::prelude::*;
use bevy_rapier2d::prelude::*;

use crate::{
    character::{CharAttrs, CharInput, Character, HorizontalDir, CHAR_WIDTH},
    hit::{Hit, HitBundle},
    micro_fighter_env::{AppState, Floor, FIXED_TIMESTEP},
};

const JUMP_VEL: f32 = 500.0;
const AIR_VEL: f32 = 2.0;
const HIT_OFFSET: Vec2 = Vec2 {
    x: CHAR_WIDTH / 2.0,
    y: 0.0,
};

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
                handle_shield_start,
                handle_shield,
                handle_shield_end,
                update_timer,
            )
                .in_set(OnUpdate(AppState::Running)),
        )
        .add_systems(
            (
                handle_move_state::<IdleState>,
                handle_move_state::<RunState>,
                handle_move_state::<JumpState>,
                handle_move_state::<FallState>,
                handle_move_state::<LightAttackStartupState>,
                handle_move_state::<LightAttackHitState>,
                handle_move_state::<LightAttackRecoveryState>,
                handle_move_state::<HeavyAttackStartupState>,
                handle_move_state::<HeavyAttackHitState>,
                handle_move_state::<HeavyAttackRecoveryState>,
                handle_move_state::<GrabState>,
                handle_move_state::<ShieldState>,
                handle_move_state::<HitstunState>,
            )
                .in_set(OnUpdate(AppState::Running)),
        );
    }
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
pub struct GrabState;

/// Holds the current frame since this state started.
#[derive(Component)]
pub struct StateTimer {
    pub frames: u32,
}

/// Treats the component as a move state.
/// Clears the state timer whenever the component is added.
fn handle_move_state<T: Component>(mut timer_query: Query<&mut StateTimer, Added<T>>) {
    for mut timer in timer_query.iter_mut() {
        timer.frames = 0;
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
        }
        else if char_inpt.shield {
            commands
                .entity(e)
                .insert(ShieldState)
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
        if timer.frames == ((char_attrs.jump_height as f32 / JUMP_VEL) / FIXED_TIMESTEP) as u32 {
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
    for (e, char_inpt, mut vel, character) in char_query.iter_mut() {
        let floor_e = floor_query.single();
        for ev in ev_collision.iter() {
            if let CollisionEvent::Started(e1, e2, _) = ev {
                let floor_collider = character.floor_collider.unwrap();
                if (*e1 == floor_collider || *e2 == floor_collider)
                    && (*e1 == floor_e || *e2 == floor_e)
                {
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
                commands.entity(hit_e).despawn();
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
                commands.entity(hit_e).despawn();
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
        let mut character = char_query.get_mut(e).unwrap();
        character.shielding = false;
    }
}
