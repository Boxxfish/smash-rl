use bevy::prelude::*;
use bevy_rapier2d::prelude::*;

use crate::{
    character::{CharAttrs, Character, HorizontalDir},
    micro_fighter_env::AppState,
};

/// Plugin for hit functionality.
pub struct HitPlugin;

impl Plugin for HitPlugin {
    fn build(&self, app: &mut App) {
        app.add_system(compute_hit_interactions.in_set(OnUpdate(AppState::Running)));
    }
}

/// Denotes that the entity is a hitbox.
#[derive(Component)]
pub struct Hit {
    pub damage: u32,
    pub direction: Vec2,
    pub chars_hit: Vec<Entity>,
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
    pub fn new(
        damage: u32,
        size: u32,
        dist: u32,
        angle: u32,
        offset: Vec2,
        dir: HorizontalDir,
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

/// Causes characters to go flying when hit.
fn compute_hit_interactions(
    mut hit_query: Query<(&mut Hit, &Parent)>,
    mut char_query: Query<&mut Character, With<Character>>,
    mut ev_collision: EventReader<CollisionEvent>,
    mut commands: Commands,
) {
    for ev in ev_collision.iter() {
        if let CollisionEvent::Started(e1, e2, _) = ev {
            let ((mut hit, hit_parent), mut character, char_e) =
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
            if hit_parent.get() == char_e {
                continue;
            }

            // If the character has already been hit, skip
            if hit.chars_hit.contains(&char_e) {
                continue;
            }

            // Apply the appropriate amount of impulse
            character.damage += hit.damage;
            let knockback =
                ((character.damage as f32) / 10.0) + (character.damage * hit.damage) as f32 / 20.0;
            let impulse = hit.direction * knockback;
            commands
                .get_entity(char_e)
                .unwrap()
                .insert(ExternalImpulse {
                    impulse,
                    ..default()
                });

            // Add character to hit's hit list
            hit.chars_hit.push(char_e);
        }
    }
}
