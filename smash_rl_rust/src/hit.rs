use bevy::prelude::*;
use bevy_rapier2d::prelude::*;

use crate::{
    character::{Character, HorizontalDir},
    micro_fighter::AppState, move_states::HitstunState,
};

/// Conversion of knockback units to velocity.
const KB_TO_VEL: f32 = 4.0;
/// Number of frames in hitstun as a percentage of knockback.
const HITSTUN_PCT: f32 = 0.4;

/// Plugin for hit functionality.
pub struct HitPlugin;

impl Plugin for HitPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            (compute_hit_interactions, move_projectile_and_remove)
                .in_set(OnUpdate(AppState::Running)),
        );
    }
}

/// The effect of the hit.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum HitType {
    /// Knocks back an opponent. Can be shielded.
    Normal,
    /// Throws an opponent. Can't be shielded.
    Grab,
}

/// Denotes that the entity is a hitbox.
#[derive(Component, Clone)]
pub struct Hit {
    pub damage: u32,
    pub direction: Vec2,
    pub chars_hit: Vec<Entity>,
    pub owner: Entity,
    pub hit_type: HitType,
}

/// Denotes a projectile.
/// Doesn't do any damage by itself.
#[derive(Component, Clone)]
pub struct Projectile {
    pub frames_left: u32,
    pub speed: u32,
    pub dir: HorizontalDir,
}

/// Bundle for hits.
#[derive(Bundle)]
pub struct HitBundle {
    pub hit: Hit,
    pub collider: Collider,
    pub transform: TransformBundle,
    pub active_events: ActiveEvents,
    pub sensor: Sensor,
}

impl HitBundle {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        damage: u32,
        size: u32,
        dist: u32,
        angle: u32,
        offset: Vec2,
        dir: HorizontalDir,
        owner: Entity,
        hit_type: HitType,
    ) -> Self {
        let dir_mult = match dir {
            HorizontalDir::Left => -1.0,
            HorizontalDir::Right => 1.0,
        };
        let mut translation = Vec2::new(
            (angle as f32).to_radians().cos() * dist as f32 / 2.0,
            (angle as f32).to_radians().sin() * dist as f32 / 2.0
                - (angle as f32).to_radians().cos() * size as f32 / 2.0,
        ) + offset;
        translation.x *= dir_mult;
        Self {
            hit: Hit {
                chars_hit: Vec::new(),
                damage,
                direction: Vec2::new(
                    (angle as f32).to_radians().cos() * dir_mult,
                    (angle as f32).to_radians().sin(),
                ),
                owner,
                hit_type,
            },
            collider: Collider::cuboid(dist as f32 / 2.0, size as f32 / 2.0),
            transform: TransformBundle::from(Transform {
                translation: translation.extend(0.0),
                rotation: Quat::from_rotation_z((angle as f32 * dir_mult).to_radians()),
                ..default()
            }),
            active_events: ActiveEvents::COLLISION_EVENTS,
            sensor: Sensor,
        }
    }
}

/// Causes a character to go in hitstun when applied.
#[derive(Component)]
pub struct Hitstun {
    /// Number of frames before recovery.
    pub frames: u32,
}

/// Causes characters to go flying when hit.
fn compute_hit_interactions(
    mut hit_query: Query<&mut Hit>,
    mut char_query: Query<&mut Character, With<Character>>,
    mut ev_collision: EventReader<CollisionEvent>,
    mut commands: Commands,
) {
    for ev in ev_collision.iter() {
        if let CollisionEvent::Started(e1, e2, _) = ev {
            let (mut hit, mut character, char_e) =
                if hit_query.get(*e1).is_ok() && char_query.get(*e2).is_ok() {
                    (
                        hit_query.get_mut(*e1).unwrap(),
                        char_query.get_mut(*e2).unwrap(),
                        *e2,
                    )
                } else if hit_query.get(*e2).is_ok() && char_query.get(*e1).is_ok() {
                    (
                        hit_query.get_mut(*e2).unwrap(),
                        char_query.get_mut(*e1).unwrap(),
                        *e1,
                    )
                } else {
                    continue;
                };

            // If the character that created the hit has been hit, skip
            if hit.owner == char_e {
                continue;
            }

            // If the character has already been hit, skip
            if hit.chars_hit.contains(&char_e) {
                continue;
            }

            // Compute normal hit
            if hit.hit_type == HitType::Normal && !character.shielding {
                // Apply the appropriate amount of impulse
                character.damage += hit.damage;
                let knockback = ((character.damage as f32) / 10.0)
                    + (character.damage * hit.damage) as f32 / 20.0;
                let impulse = hit.direction * knockback * KB_TO_VEL;
                let hitstun_frames = (knockback * HITSTUN_PCT) as u32;
                commands
                    .get_entity(char_e)
                    .unwrap()
                    .insert(ExternalImpulse {
                        impulse,
                        ..default()
                    })
                    .insert(HitstunState)
                    .insert(Hitstun {
                        frames: hitstun_frames,
                    });
            }

            // Compute grab
            if hit.hit_type == HitType::Grab {
                // Apply the appropriate amount of impulse
                character.damage += hit.damage;
                let knockback = ((character.damage as f32) / 10.0)
                    + (character.damage * hit.damage) as f32 / 20.0;
                let impulse = hit.direction * knockback * KB_TO_VEL;
                let hitstun_frames = (knockback * HITSTUN_PCT) as u32;
                commands
                    .get_entity(char_e)
                    .unwrap()
                    .insert(ExternalImpulse {
                        impulse,
                        ..default()
                    })
                    .insert(HitstunState)
                    .insert(Hitstun {
                        frames: hitstun_frames,
                    });
            }

            // Add character to hit's hit list
            hit.chars_hit.push(char_e);
        }
    }
}

/// Moves the projectile and removes it when time runs out.
fn move_projectile_and_remove(
    mut proj_query: Query<(Entity, &mut Projectile, &mut Transform)>,
    mut commands: Commands,
) {
    for (e, mut projectile, mut transform) in proj_query.iter_mut() {
        // Move the projectile forward
        let dir = match projectile.dir {
            HorizontalDir::Left => -Vec3::X,
            HorizontalDir::Right => Vec3::X,
        };
        transform.translation += dir * projectile.speed as f32;

        // Remove projectile when time runs out
        projectile.frames_left = projectile.frames_left.saturating_sub(1);
        if projectile.frames_left == 0 {
            commands.entity(e).despawn();
        }
    }
}
