use bevy::{math::Vec3Swizzles, prelude::*};
use bevy_rapier2d::prelude::*;
use pyo3::prelude::*;

use crate::{
    character::{Bot, CharAttrs, CharBundle, CharInput, Character, Player, CHAR_WIDTH},
    hit::{Hit, Projectile},
    micro_fighter::{
        AppState, OPPONENT_COLL_FILTER, OPPONENT_COLL_GROUP, PLAYER_COLL_FILTER, PLAYER_COLL_GROUP,
    },
    move_states::{
        GrabState, HeavyAttackRecoveryState, HeavyAttackStartupState, HitstunState,
        LightAttackRecoveryState, LightAttackStartupState, ShieldState, StateTimer, MoveState, CurrentMoveState, add_move_state,
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
            .insert_resource(GameState::default())
            .configure_set(
                MLBaseSet::MLWork
                    .after(CoreSet::Update)
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
    for (
        glob_transform,
        character,
        collider,
        player,
        curr_move_state,
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
    pub player_state: Option<CharGameState>,
    pub opponent_state: Option<CharGameState>,
    pub hits: Vec<HitState>,
}

/// Stores the current state of a character.
#[derive(Clone)]
pub struct CharGameState {
    pub pos: Vec2,
    pub vel: Vec2,
    // This will have to be manually computed
    // pub acc: Vec2,
    pub damage: u32,
    pub state: MoveState,
    pub attrs: CharAttrs,
    pub frame_counter: u32,
}

/// Stores the state of a hit.
#[derive(Clone)]
pub struct HitState {
    pub hit: Hit,
    pub projectile: Option<Projectile>,
    pub pos: Vec2,
    pub extents: Vec2,
}

/// Items in the character query when saving state.
type CharQueryItems<'a> = (
    &'a GlobalTransform,
    &'a Character,
    &'a CharAttrs,
    &'a Velocity,
    &'a StateTimer,
    &'a CurrentMoveState,
);

/// Updates the current game state resource.
fn update_game_state(
    mut game_state: ResMut<GameState>,
    char_query: Query<CharQueryItems>,
    player_query: Query<Entity, With<Player>>,
    bot_query: Query<Entity, With<Bot>>,
    hit_query: Query<(&Hit, &Transform, &Collider, Option<&Projectile>)>,
) {
    // Store character data
    let player_state = extract_char_state(player_query.single(), &char_query);
    let opponent_state = extract_char_state(bot_query.single(), &char_query);

    // Store all hits
    let mut hits = Vec::new();
    for (hit, transform, collider, projectile) in hit_query.iter() {
        let hit_state = HitState {
            hit: hit.clone(),
            projectile: projectile.cloned(),
            pos: transform.translation.xy(),
            extents: collider.as_cuboid().unwrap().half_extents(),
        };
        hits.push(hit_state);
    }

    game_state.player_state = Some(player_state);
    game_state.opponent_state = Some(opponent_state);
    game_state.hits = hits;
}

/// Helper function for getting character state from a query.
fn extract_char_state(char_e: Entity, char_query: &Query<CharQueryItems>) -> CharGameState {
    let (
        glob_transform,
        character,
        attrs,
        velocity,
        state_timer,
        curr_move_state,
    ) = char_query.get(char_e).unwrap();

    let pos = glob_transform.translation().xy();
    let vel = velocity.linvel;
    let damage = character.damage;
    let attrs = *attrs;
    let frame_counter = state_timer.frames;
    let state = curr_move_state.move_state;

    CharGameState {
        pos,
        vel,
        damage,
        state,
        attrs,
        frame_counter,
    }
}

/// Causes the game to load the state given.
pub struct LoadStateEvent {
    pub game_state: GameState,
}

/// Loads the game state.
fn load_game_state(
    mut ev_load_state: EventReader<LoadStateEvent>,
    player_query: Query<Entity, With<Player>>,
    bot_query: Query<Entity, With<Bot>>,
    mut commands: Commands,
) {
    for ev in ev_load_state.iter() {
        let (player_e, bot_e) = if !player_query.is_empty() {
            // Remove player and bot children if they exist
            let player_e = player_query.single();
            let bot_e = bot_query.single();
            commands.entity(player_e).clear_children();
            commands.entity(bot_e).clear_children();
            (player_e, bot_e)
        } else {
            // Othewise, create new player and bot
            (
                commands.spawn(Player::default()).id(),
                commands.spawn(Bot).id(),
            )
        };

        // Add player
        let player_state = ev.game_state.player_state.as_ref().unwrap();
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
            transform: TransformBundle::from(Transform::from_xyz(
                player_state.pos.x,
                player_state.pos.y,
                0.0,
            )),
            ..default()
        };
        p_bundle.character.floor_collider = Some(p_floor_collider);
        commands
            .entity(player_e)
            .insert(p_bundle)
            .add_child(p_floor_collider);
        add_move_state(player_state.state, player_e, &mut commands);

        // Add bot
        let bot_state = ev.game_state.opponent_state.as_ref().unwrap();
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
            transform: TransformBundle::from(Transform::from_xyz(
                bot_state.pos.x,
                bot_state.pos.y,
                0.0,
            )),
            ..default()
        };
        b_bundle.character.floor_collider = Some(b_floor_collider);
        commands
            .entity(bot_e)
            .insert(b_bundle)
            .add_child(b_floor_collider);
        add_move_state(bot_state.state, bot_e, &mut commands);

        // Add hits
        for hit_state in &ev.game_state.hits {
            let transform =
                TransformBundle::from(Transform::from_translation(hit_state.pos.extend(0.0)));
            let collider = Collider::cuboid(hit_state.extents.x, hit_state.extents.y);
            let hit_e = commands
                .spawn((hit_state.hit.clone(), collider, transform))
                .id();
            if hit_state.projectile.is_none() {
                commands.entity(hit_state.hit.owner).add_child(hit_e);
            } else {
                commands
                    .entity(hit_e)
                    .insert(hit_state.projectile.clone().unwrap());
            }
        }
    }
}
