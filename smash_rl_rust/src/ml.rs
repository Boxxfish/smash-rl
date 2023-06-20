use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use pyo3::prelude::*;

use crate::{
    character::{Bot, CharAttrs, CharInput, Character, Player, CHAR_WIDTH},
    hit::Hit,
    micro_fighter::AppState,
    move_states::{
        GrabState, HeavyAttackRecoveryState, HeavyAttackStartupState, HitstunState,
        LightAttackRecoveryState, LightAttackStartupState, ShieldState,
    },
};

/// Plugin for systems required for ML.
pub struct MLPlugin;

impl Plugin for MLPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MinimalPlugins)
            .add_plugin(TransformPlugin)
            .add_plugin(HierarchyPlugin)
            .insert_resource(HBoxCollection::default())
            .add_systems(
                (handle_ml_player_input, collect_hboxes).in_set(OnUpdate(AppState::Running)),
            )
            .add_event::<MLPlayerActionEvent>();
    }
}

/// Information for both hitboxes and hurtboxes.
/// These are rotated boxes.
#[pyclass]
#[derive(Clone)]
pub struct HBox {
    /// Type of box. If true, this is a hitbox. Otherwise, this is a hurtbox.
    #[pyo3(get)]
    pub is_hit: bool,
    /// X coordinate of top left corner.
    #[pyo3(get)]
    pub x: i32,
    /// Y coordinate of top left corner.
    #[pyo3(get)]
    pub y: i32,
    /// Width of box.
    #[pyo3(get)]
    pub w: u32,
    /// Height of box.
    #[pyo3(get)]
    pub h: u32,
    /// Angle of the box in radians.
    #[pyo3(get)]
    pub angle: f32,
    /// Whether this box belongs to the player or the opponent.
    #[pyo3(get)]
    pub is_player: bool,
    /// Damage of the character if this is a hurtbox, or damage the hitbox will
    /// inflict.
    #[pyo3(get)]
    pub damage: u32,
    /// State the character currently is in.
    #[pyo3(get)]
    pub char_state: CharState,
}

/// State of the character.
/// Removes some of the partial observiability.
#[pyclass]
#[derive(Copy, Clone)]
pub enum CharState {
    StartupHeavy,
    RecoveryHeavy,
    StartupLight,
    RecoveryLight,
    Grab,
    Shield,
    Hitstun,
    Other,
}

/// Resource that stores all HBoxes.
#[derive(Resource, Default)]
pub struct HBoxCollection {
    pub hboxes: Vec<HBox>,
}

/// Updates the HBoxCollection.
#[allow(clippy::type_complexity)]
fn collect_hboxes(
    char_query: Query<(
        &GlobalTransform,
        &Character,
        &Collider,
        Option<&Player>,
        Option<&HeavyAttackStartupState>,
        Option<&HeavyAttackRecoveryState>,
        Option<&LightAttackStartupState>,
        Option<&LightAttackRecoveryState>,
        Option<&GrabState>,
        Option<&ShieldState>,
        Option<&HitstunState>,
    )>,
    hit_query: Query<(&Hit, &GlobalTransform, &Collider)>,
    player_query: Query<Entity, With<Player>>,
    mut hbox_coll: ResMut<HBoxCollection>,
) {
    hbox_coll.hboxes.clear();

    // Collect hurtboxes
    let player_e = player_query.single();
    let mut player_state = None;
    let mut opp_state = None;
    for (
        glob_transform,
        character,
        collider,
        player,
        heavy_startup,
        heavy_recovery,
        light_startup,
        light_recovery,
        grab,
        shield,
        hitstun,
    ) in char_query.iter()
    {
        let transform = glob_transform.compute_transform();
        let collider = collider.as_cuboid().unwrap();
        let x = (transform.translation.x - collider.half_extents().x) as i32;
        let y = (transform.translation.y - collider.half_extents().y) as i32;
        let w = (collider.half_extents().x * 2.0) as u32;
        let h = (collider.half_extents().y * 2.0) as u32;
        let angle = 0.0;
        let is_player = player.is_some();
        let damage = character.damage;
        let char_state = if heavy_startup.is_some() {
            CharState::StartupHeavy
        } else if heavy_recovery.is_some() {
            CharState::RecoveryHeavy
        } else if light_startup.is_some() {
            CharState::StartupLight
        } else if light_recovery.is_some() {
            CharState::RecoveryLight
        } else if grab.is_some() {
            CharState::Grab
        } else if shield.is_some() {
            CharState::Shield
        } else if hitstun.is_some() {
            CharState::Hitstun
        } else {
            CharState::Other
        };
        if is_player {
            player_state = Some(char_state);
        } else {
            opp_state = Some(char_state);
        }
        let hbox = HBox {
            is_hit: false,
            x,
            y,
            w,
            h,
            angle,
            is_player,
            damage,
            char_state,
        };
        hbox_coll.hboxes.push(hbox);
    }

    // Collect hitboxes
    for (hit, glob_transform, collider) in hit_query.iter() {
        let transform = glob_transform.compute_transform();
        let collider = collider.as_cuboid().unwrap();
        let x = (transform.translation.x - collider.half_extents().x) as i32;
        let y = (transform.translation.y - collider.half_extents().y) as i32;
        let w = (collider.half_extents().x * 2.0) as u32;
        let h = (collider.half_extents().y * 2.0) as u32;
        let angle = transform.rotation.to_euler(EulerRot::XYZ).2;
        let is_player = hit.owner == player_e;
        let char_state = if is_player { player_state } else { opp_state }.unwrap();
        let hbox = HBox {
            is_hit: true,
            x,
            y,
            w,
            h,
            angle,
            is_player,
            damage: hit.damage,
            char_state,
        };
        hbox_coll.hboxes.push(hbox);
    }
}

/// Event indicating the ML player has taken an action.
/// 0. Do nothing.
/// 1. Left.
/// 2. Right.
/// 3. Jump.
/// 4. Light.
/// 5. Heavy.
/// 6. Shield.
/// 7. Grab.
pub struct MLPlayerActionEvent {
    pub action_id: u32,
}

/// Processes the ML player's input.
fn handle_ml_player_input(
    mut ev_ml_player_action: EventReader<MLPlayerActionEvent>,
    mut player_query: Query<&mut CharInput, With<Player>>,
) {
    // Reset player input
    let mut player_inpt = player_query.single_mut();
    player_inpt.left = false;
    player_inpt.right = false;
    player_inpt.jump = false;
    player_inpt.light = false;
    player_inpt.heavy = false;
    player_inpt.shield = false;
    player_inpt.grab = false;

    for ev in ev_ml_player_action.iter() {
        match ev.action_id {
            0 => (),
            1 => player_inpt.left = true,
            2 => player_inpt.right = true,
            3 => player_inpt.jump = true,
            4 => player_inpt.light = true,
            5 => player_inpt.heavy = true,
            6 => player_inpt.shield = true,
            7 => player_inpt.grab = true,
            _ => unreachable!(),
        }
    }
}
