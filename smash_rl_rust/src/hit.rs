use bevy::prelude::*;
use bevy_rapier2d::prelude::*;

use crate::character::HorizontalDir;

/// Plugin for hit functionality.
pub struct HitPlugin;

impl Plugin for HitPlugin {
    fn build(&self, app: &mut App) {
        
    }
}


/// Denotes that the entity is a hitbox.
#[derive(Component)]
pub struct Hit {
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
    pub fn new(size: u32, dist: u32, angle: u32, offset: Vec2, dir: HorizontalDir) -> Self {
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