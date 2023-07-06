use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use bevy_save::{AppSaveableExt, SavePlugins, WorldSaveableExt};
use pyo3::prelude::*;

use crate::{
    character::{Bot, CharAttrs, CharInput, Character, Player},
    hit::{Hit, Hitstun, Projectile},
    micro_fighter::AppState,
    move_states::{CurrentMoveState, MoveState},
};

/// Plugin for systems required for ML.
pub struct MLPlugin;

impl Plugin for MLPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MinimalPlugins)
            .add_plugin(TransformPlugin)
            .add_plugin(HierarchyPlugin)
            .add_plugins(SavePlugins)
            .insert_resource(HBoxCollection::default())
            .insert_resource(GameState::default())
            .configure_set(
                MLBaseSet::MLWork
                    .after(CoreSet::PostUpdate)
                    .run_if(in_state(AppState::Running)),
            )
            .add_systems(
                (
                    handle_ml_player_input,
                    handle_ml_bot_input,
                    collect_hboxes,
                    update_game_state,
                    load_game_state,
                )
                    .in_base_set(MLBaseSet::MLWork),
            )
            .register_type::<Option<Entity>>()
            .add_event::<MLPlayerActionEvent>()
            .add_event::<MLBotActionEvent>()
            .add_event::<LoadStateEvent>();
    }
}

/// Base set for ML stuff.
#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
#[system_set(base)]
enum MLBaseSet {
    /// Runs after CoreSet::Update.
    MLWork,
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
    pub move_state: MoveState,
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
        &CurrentMoveState,
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
    for (glob_transform, character, collider, player, curr_move_state) in char_query.iter() {
        let transform = glob_transform.compute_transform();
        let collider = collider.as_cuboid().unwrap();
        let x = (transform.translation.x - collider.half_extents().x) as i32;
        let y = (transform.translation.y - collider.half_extents().y) as i32;
        let w = (collider.half_extents().x * 2.0) as u32;
        let h = (collider.half_extents().y * 2.0) as u32;
        let angle = 0.0;
        let is_player = player.is_some();
        let damage = character.damage;
        let move_state = curr_move_state.move_state;
        if is_player {
            player_state = Some(move_state);
        } else {
            opp_state = Some(move_state);
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
            move_state,
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
        let move_state = if is_player { player_state } else { opp_state }.unwrap();
        let hbox = HBox {
            is_hit: true,
            x,
            y,
            w,
            h,
            angle,
            is_player,
            damage: hit.damage,
            move_state,
        };
        hbox_coll.hboxes.push(hbox);
    }
}

/// Event indicating the ML player has taken an action.
/// See gym definition for what action_id corresponds to.
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
    player_inpt.special = false;

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
            8 => player_inpt.special = true,
            _ => unreachable!(),
        }
    }
}

/// Event indicating the ML bot has taken an action.
/// See gym definition for what action_id corresponds to.
pub struct MLBotActionEvent {
    pub action_id: u32,
}

/// Process the ML bot's input.
fn handle_ml_bot_input(
    mut ev_ml_bot_action: EventReader<MLBotActionEvent>,
    mut bot_query: Query<&mut CharInput, With<Bot>>,
) {
    // Reset bot input
    let mut bot_inpt = bot_query.single_mut();
    bot_inpt.left = false;
    bot_inpt.right = false;
    bot_inpt.jump = false;
    bot_inpt.light = false;
    bot_inpt.heavy = false;
    bot_inpt.shield = false;
    bot_inpt.grab = false;
    bot_inpt.special = false;

    for ev in ev_ml_bot_action.iter() {
        match ev.action_id {
            0 => (),
            1 => bot_inpt.left = true,
            2 => bot_inpt.right = true,
            3 => bot_inpt.jump = true,
            4 => bot_inpt.light = true,
            5 => bot_inpt.heavy = true,
            6 => bot_inpt.shield = true,
            7 => bot_inpt.grab = true,
            8 => bot_inpt.special = true,
            _ => unreachable!(),
        }
    }
}

/// Stores the current state of the game.
/// Useful for loading and saving for MCTS.
#[pyclass]
#[derive(Resource, Default, Clone)]
pub struct GameState {
    pub ser_snapshot: Option<Vec<u8>>,
}

/// Stores the current state of a character.
#[derive(Clone)]
pub struct CharGameState {
    pub pos: Vec2,
    pub vel: Vec2,
    pub damage: u32,
    pub state: MoveState,
    pub attrs: CharAttrs,
    pub frame_counter: u32,
    pub hitstun: Option<Hitstun>,
}

/// Stores the state of a hit.
#[derive(Clone)]
pub struct HitState {
    pub hit: Hit,
    pub projectile: Option<Projectile>,
    pub pos: Vec2,
    pub extents: Vec2,
}

/// Updates the current game state resource.
fn update_game_state(world: &mut World) {
    let mut buf = Vec::new();
    world
        .serialize(&mut rmp_serde::Serializer::new(&mut buf))
        .unwrap();
    let mut res = world.get_resource_mut::<GameState>();
    let game_state = res.as_mut().unwrap();
    game_state.ser_snapshot = Some(buf);
}

/// Causes the game to load the state given.
pub struct LoadStateEvent {
    pub game_state: GameState,
}

/// Loads the game state.
fn load_game_state(world: &mut World) {
    let events = world.get_resource::<Events<LoadStateEvent>>().unwrap();
    let mut reader = events.get_reader();
    let mut snapshots = Vec::new();
    for ev in reader.iter(events) {
        if ev.game_state.ser_snapshot.is_some() {
            snapshots.push(ev.game_state.ser_snapshot.as_ref().cloned().unwrap());
        } else {
            eprint!("Attempted to load nonexistant state. Ignoring.");
        }
    }
    for snapshot in &snapshots {
        world
            .deserialize(&mut rmp_serde::Deserializer::new(snapshot.as_slice()))
            .unwrap();
    }
}
