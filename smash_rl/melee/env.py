import time
from typing import Any, Optional, SupportsFloat
from gymnasium import Env
import gymnasium as gym
import numpy as np
import melee  # type: ignore
from melee import Button, Action
import random
from gymnasium.wrappers.time_limit import TimeLimit
import pygame
import os

IMG_SIZE = 32
IMG_SCALE = 8
X_MIN, X_MAX, Y_MAX, Y_MIN = melee.BLASTZONES[melee.Stage.FINAL_DESTINATION]

ACTION_MAP = {
    # Idle
    Action.STANDING: 0,
    # Walking
    Action.WALK_SLOW: 1,
    Action.WALK_MIDDLE: 1,
    Action.WALK_FAST: 1,
    # Running
    Action.RUNNING: 2,
    Action.DASHING: 2,
    # Crouching
    Action.CROUCH_START: 3,
    Action.CROUCHING: 3,
    Action.CROUCH_END: 3,
    # Neutral attack
    Action.NEUTRAL_ATTACK_1: 4,
    Action.NEUTRAL_ATTACK_2: 4,
    Action.NEUTRAL_ATTACK_3: 4,
    Action.LOOPING_ATTACK_START: 4,
    Action.LOOPING_ATTACK_MIDDLE: 4,
    Action.LOOPING_ATTACK_END: 4,
    # Dash attack
    Action.DASH_ATTACK: 5,
    # FTilt
    Action.FTILT_HIGH: 6,
    Action.FTILT_HIGH_MID: 6,
    Action.FTILT_MID: 6,
    Action.FTILT_LOW_MID: 6,
    Action.FTILT_LOW: 6,
    # DTilt
    Action.DOWNTILT: 7,
    # UTilt
    Action.UPTILT: 8,
    # FSmash
    Action.FSMASH_HIGH: 9,
    Action.FSMASH_MID_HIGH: 9,
    Action.FSMASH_MID: 9,
    Action.FSMASH_MID_LOW: 9,
    Action.FSMASH_LOW: 9,
    # DSmash
    Action.DOWNSMASH: 10,
    # USmash
    Action.UPSMASH: 11,
    # FAir
    Action.FAIR: 12,
    Action.FAIR_LANDING: 12,
    # BAir
    Action.BAIR: 13,
    Action.BAIR_LANDING: 13,
    # Neutral B
    Action.NEUTRAL_B_ATTACKING: 14,
    Action.NEUTRAL_B_ATTACKING_AIR: 14,
    Action.NEUTRAL_B_CHARGING: 14,
    Action.NEUTRAL_B_CHARGING_AIR: 14,
    Action.NEUTRAL_B_FULL_CHARGE: 14,
    Action.NEUTRAL_B_FULL_CHARGE_AIR: 14,
    Action.LASER_GUN_PULL: 14,
    # Up B
    Action.UP_B_AIR: 15,
    Action.UP_B_GROUND: 15,
    Action.KIRBY_BLADE_APEX: 15,
    Action.KIRBY_BLADE_DOWN: 15,
    Action.KIRBY_BLADE_GROUND: 15,
    # Down B
    Action.DOWN_B_GROUND_START: 16,
    Action.DOWN_B_GROUND: 16,
    Action.DOWN_B_AIR: 16,
    Action.KIRBY_STONE_UNFORMING: 16,
    Action.KIRBY_STONE_FORMING_AIR: 16,
    Action.KIRBY_STONE_FORMING_GROUND: 16,
    Action.KIRBY_STONE_RELEASE: 16,
    Action.KIRBY_STONE_RESTING: 16,
    # Side B
    Action.FOX_ILLUSION: 17,
    Action.FOX_ILLUSION_SHORTENED: 17,
    Action.FOX_ILLUSION_START: 17,
    # Shield
    Action.SHIELD: 18,
    Action.SHIELD_RELEASE: 18,
    Action.SHIELD_START: 18,
    Action.SHIELD_STUN: 18,
    # Dodge
    Action.AIRDODGE: 19,
    Action.SPOTDODGE: 19,
    # Roll
    Action.ROLL_BACKWARD: 20,
    Action.ROLL_FORWARD: 20,
    Action.EDGE_ROLL_QUICK: 20,
    Action.EDGE_ROLL_SLOW: 20,
    # DAir
    Action.DAIR: 21,
    # Air
    Action.FALLING: 22,
    Action.LANDING: 22,
    # UAir
    Action.UAIR: 23,
    Action.UAIR_LANDING: 23,
    # Can't move
    Action.DEAD_FALL: 24,
    Action.DEAD_DOWN: 24,
    Action.GRABBED: 24,
    Action.GRAB_PUMMELED: 24,
    Action.DAMAGE_AIR_1: 24,
    Action.DAMAGE_AIR_2: 24,
    Action.DAMAGE_AIR_3: 24,
    Action.DAMAGE_BIND: 24,
    Action.DAMAGE_NEUTRAL_1: 24,
    Action.DAMAGE_HIGH_1: 24,
    Action.DAMAGE_HIGH_2: 24,
    Action.DAMAGE_HIGH_3: 24,
    Action.DAMAGE_SONG: 24,
    Action.DAMAGE_GROUND: 24,
    Action.DAMAGE_ICE: 24,
    Action.DAMAGE_SCREW: 24,
    Action.DAMAGE_SCREW_AIR: 24,
    # Grab
    Action.GRAB: 25,
    Action.GRAB_PUMMEL: 25,
    # Throw
    Action.THROW_BACK: 26,
    Action.THROW_DOWN: 26,
    Action.THROW_FORWARD: 26,
    Action.THROW_UP: 26,
    # Edge
    Action.EDGE_HANGING: 27,
    # Roll
    Action.EDGE_ROLL_QUICK: 28,
    Action.EDGE_ROLL_SLOW: 28,
    Action.ROLL_FORWARD: 28,
    Action.ROLL_BACKWARD: 28,
}
STATE_COUNT = max(ACTION_MAP.values()) + 1

PROJECTILE_MAP = {
    melee.ProjectileType.FOX_LASER: {
        "w": 16,
        "h": 4,
    }
}


class MeleeEnv(Env):
    """
    Environment that wraps a Dolphin emulator.
    The agent is randomly assigned as player 1 or 2.
    """

    def __init__(
        self,
        console_id: int,
        view_channels: tuple[int, int, int] = (0, 1, 2),
        render_mode: Optional[str] = None,
        num_frames: int = 4,
    ):
        self.console = melee.Console(
            path=os.environ["HOME"] + "/slippi_dolphin/usr/bin/dolphin-emu",
            fullscreen=False,
            disable_audio=True,
            slippi_port=51441 + console_id,
            system="dolphin",
            blocking_input=True,
        )
        self.console.setup_dolphin_controller(1)
        self.console.setup_dolphin_controller(2)
        self.console.run(iso_path=os.environ["HOME"] + "/DietMeleeClassic_1.03.iso")
        self.console.connect()
        self.num_channels = 4
        self.single_stats_count = 4 + STATE_COUNT
        self.observation_space = gym.spaces.Tuple(
            [
                gym.spaces.Box(
                    -1.0, 1.0, [num_frames, self.num_channels, IMG_SIZE, IMG_SIZE]
                ),
                gym.spaces.Box(-1.0, 2.0, [(self.single_stats_count) * 2]),
            ]
        )
        self.action_space = gym.spaces.Discrete(
            1 + 5 * 7 + 4
        )  # Null action, cartesian set of buttons and main stick directions, and c stick directions
        self.pads = [
            melee.Controller(self.console, port=1),
            melee.Controller(self.console, port=2),
        ]
        for pad in self.pads:
            pad.connect()
        self.player_pad = 0
        self.button_list = [
            Button.BUTTON_A,
            Button.BUTTON_B,
            Button.BUTTON_X,
            Button.BUTTON_L,
            Button.BUTTON_Z,
        ]
        self.main_stick_dir = [
            (0.5, 0.5),  # Center
            (0.5, 1.0),  # Up
            (0.5, -1.0),  # Down
            (-1.0, 0.5),  # Run left
            (-0.75, 0.5),  # Walk left
            (1.0, 0.5),  # Run right
            (-0.75, 0.5),  # Walk right
        ]
        self.c_stick_dir = [
            (0.5, 0.5),  # Center
            (0.5, 1.0),  # Up
            (0.5, -1.0),  # Down
            (-1.0, 0.5),  # Left
            (1.0, 0.5),  # Right
        ]
        self.last_buttons: list[Optional[int]] = [None, None]
        self.last_stocks1 = 0
        self.last_stocks2 = 0
        self.framedata = melee.FrameData()
        self.bot_action = 0

        # Obs stuff
        self.num_frames = num_frames
        self.player_frame_stack = [
            np.zeros([self.num_channels, IMG_SIZE, IMG_SIZE])
            for _ in range(self.num_frames)
        ]
        self.bot_frame_stack = [
            np.zeros([self.num_channels, IMG_SIZE, IMG_SIZE])
            for _ in range(self.num_frames)
        ]
        self.player_stats = np.zeros([self.single_stats_count])
        self.bot_stats = np.zeros([self.single_stats_count])

        self.last_player_percent = 0.0
        self.last_bot_percent = 0.0

        self.render_mode = render_mode
        self.view_channels = view_channels
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (IMG_SIZE * IMG_SCALE, IMG_SIZE * IMG_SCALE)
            )
            self.clock = pygame.time.Clock()

    def step(
        self, action: int
    ) -> tuple[tuple[np.ndarray, np.ndarray], float, bool, bool, dict[str, Any]]:
        # Step and get gamestate
        gamestate = None
        frames_skipped = 0
        while gamestate is None or gamestate.menu_state is not melee.Menu.IN_GAME or frames_skipped < 2:
            self.process_input(self.bot_action, 1 - self.player_pad)
            self.process_input(action, self.player_pad)
            gamestate = self.console.step()
            frames_skipped += 1
        # time.sleep(0.1)

        # Handle losing a stock
        done = False
        reward = 0.0
        stocks1 = int(gamestate.players[1].stock)
        stocks2 = int(gamestate.players[2].stock)
        diff_stocks1 = stocks1 - self.last_stocks1
        diff_stocks2 = stocks2 - self.last_stocks2
        self.last_stocks1 = stocks1
        self.last_stocks2 = stocks2

        if diff_stocks1 == -1:
            if self.player_pad == 0:
                reward += -1.0
            else:
                reward += 1.0

        if diff_stocks2 == -1:
            if self.player_pad == 1:
                reward += -1.0
            else:
                reward += 1.0

        info = {}
        if diff_stocks1 == -1 or diff_stocks2 == -1:
            done = True
            info["player_won"] = (diff_stocks2 == -1 and self.player_pad == 0) or (
                diff_stocks1 == -1 and self.player_pad == 1
            )

        # Add exploration reward
        player_percent = gamestate.players[self.player_pad + 1].percent
        bot_percent = gamestate.players[2 - self.player_pad].percent
        diff_player = player_percent - self.last_player_percent # We want this to decrease
        diff_bot = bot_percent - self.last_bot_percent # We want this to increase
        reward += (-diff_player + diff_bot) / 100.0
        self.last_player_percent = player_percent
        self.last_bot_percent = bot_percent

        # Generate stats
        self.player_stats = compute_stats_single(
            gamestate, self.player_pad, self.framedata
        )
        self.bot_stats = compute_stats_single(
            gamestate, 1 - self.player_pad, self.framedata
        )

        # Generate spatial
        channels = self.gen_channels(gamestate, player_id=self.player_pad)
        self.player_frame_stack = self.insert_obs(
            np.stack(channels), self.player_frame_stack
        )
        channels = self.gen_channels(gamestate, player_id=1 - self.player_pad)
        self.bot_frame_stack = self.insert_obs(np.stack(channels), self.bot_frame_stack)

        return (
            (
                np.stack(self.player_frame_stack),
                np.concatenate([self.player_stats, self.bot_stats]),
            ),
            reward,
            done,
            False,
            info,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
        # if self.console.connected:
        #     self.console._slippstream.shutdown()
        #     self.console.connected = False
        #     self.console.connect()
        self.pads[0].release_all()
        self.pads[1].release_all()
        self.pads[0].flush()
        self.pads[1].flush()

        # Reload state
        self.pads[0].press_button(Button.BUTTON_D_DOWN)
        self.pads[0].flush()
        time.sleep(0.1)
        self.pads[0].release_button(Button.BUTTON_D_DOWN)
        self.pads[0].flush()
        time.sleep(1.0)

        # Hold down speed delimiter

        # Release speed delimiter
        # self.pads[0].release_all()
        # self.console.step()

        self.player_pad = random.randrange(0, 2)
        self.bot_action = 0

        # Generate stats
        gamestate = None
        while gamestate is None or gamestate.menu_state is not melee.Menu.IN_GAME:
            gamestate = self.console.step()
        self.player_stats = compute_stats_single(
            gamestate, self.player_pad, self.framedata
        )
        self.bot_stats = compute_stats_single(
            gamestate, 1 - self.player_pad, self.framedata
        )
        self.last_stocks1 = int(gamestate.players[1].stock)
        self.last_stocks2 = int(gamestate.players[2].stock)

        # Generate spatial
        self.player_frame_stack = [
            np.zeros([self.num_channels, IMG_SIZE, IMG_SIZE])
            for _ in range(self.num_frames)
        ]
        self.bot_frame_stack = [
            np.zeros([self.num_channels, IMG_SIZE, IMG_SIZE])
            for _ in range(self.num_frames)
        ]

        self.last_player_percent = 0.0
        self.last_bot_percent = 0.0
        self.last_buttons = [None, None]

        return (
            np.stack(self.player_frame_stack),
            np.concatenate([self.player_stats, self.bot_stats]),
        ), {}

    def process_input(self, action: int, pad_id: int):
        if action == 0:
            self.pads[pad_id].release_all()
        elif action < 1 + 5 * 7:
            action_idx = action - 1
            button_idx = action_idx // 7

            last_button = self.last_buttons[pad_id]
            if last_button is not button_idx:
                self.pads[pad_id].press_button(self.button_list[button_idx])
                if last_button is not None:
                    self.pads[pad_id].release_button(self.button_list[last_button])
            self.last_buttons[pad_id] = button_idx

            dir_idx = action_idx - button_idx * 7
            main_stick_dir = self.main_stick_dir[dir_idx]
            self.pads[pad_id].tilt_analog(
                Button.BUTTON_MAIN, main_stick_dir[0], main_stick_dir[1]
            )
        else:
            self.pads[pad_id].release_all()
            c_idx = action - (1 + 5 * 7)
            c_stick_dir = self.c_stick_dir[c_idx]
            self.pads[pad_id].tilt_analog(
                Button.BUTTON_C, c_stick_dir[0], c_stick_dir[1]
            )

    def render(self):
        if self.render_mode == "human":
            channels = self.player_frame_stack[0]#self.player_frame_stack[0]
            r = channels[self.view_channels[0]]
            g = channels[self.view_channels[1]]
            b = channels[self.view_channels[2]]
            view = np.flip(np.stack([r, g, b]).transpose(2, 1, 0), 1).clip(0, 1) * 255.0
            view_surf = pygame.Surface([IMG_SIZE, IMG_SIZE])
            pygame.surfarray.blit_array(view_surf, view)
            pygame.transform.scale(
                view_surf, [IMG_SIZE * IMG_SCALE, IMG_SIZE * IMG_SCALE], self.screen
            )
            pygame.display.flip()
            # self.clock.tick(60)

    def disconnect(self):
        """
        Disconnects from the simulation.
        If the env will be paused for more than a couple seconds, this should be called.
        """
        self.console._slippstream.shutdown()

    def reconnect(self):
        """
        Reconnects to the simulation.
        Required after disconnecting.
        """
        self.console._slippstream.connect()

    def gen_channels(
        self, gamestate: melee.GameState, player_id: int
    ) -> list[np.ndarray]:
        """
        Converts `gamestate` into observation channels for a player.
        """
        # x_center = (gamestate.players[1].position.x + gamestate.players[2].position.x) / 2
        # y_center = (gamestate.players[1].position.y + gamestate.players[2].position.y) / 2
        # x_bound = max(abs(gamestate.players[1].position.x - x_center), abs(gamestate.players[2].position.x - x_center))
        # y_bound = max(abs(gamestate.players[1].position.y - y_center), abs(gamestate.players[2].position.y - y_center))
        # max_bound = min(max(max(x_bound, y_bound) + 10, 50), 300)
        # global X_MIN, X_MAX, Y_MIN, Y_MAX
        # X_MIN = x_center - max_bound
        # X_MAX = x_center + max_bound
        # Y_MIN = y_center - max_bound
        # Y_MAX = y_center + max_bound

        # Create stage channel
        stage = gamestate.stage
        stage_channel = np.zeros([IMG_SIZE, IMG_SIZE])

        stage_r = melee.stages.EDGE_POSITION[stage]
        stage_l = -stage_r
        stage_y = 0.0
        stage_l = view_space((stage_l, stage_y), IMG_SIZE)
        stage_r = view_space((stage_r, stage_y), IMG_SIZE)
        for x in range(stage_l[0], stage_r[0]):
            stage_channel[stage_l[1]][x] = 1

        top_y, top_l, top_r = melee.stages.top_platform_position(stage)
        if top_y:
            top_l = view_space((top_l, top_y), IMG_SIZE)
            top_r = view_space((top_r, top_y), IMG_SIZE)
            for x in range(top_l[0], top_r[0]):
                stage_channel[top_l[1]][x] = 1

        left_y, left_l, left_r = melee.stages.left_platform_position(stage)
        if left_y:
            left_l = view_space((left_l, left_y), IMG_SIZE)
            left_r = view_space((left_r, left_y), IMG_SIZE)
            for x in range(left_l[0], left_r[0]):
                stage_channel[left_l[1]][x] = 1

        right_y, right_l, right_r = melee.stages.right_platform_position(stage)
        if right_y:
            right_l = view_space((right_l, right_y), IMG_SIZE)
            right_r = view_space((right_r, right_y), IMG_SIZE)
            for x in range(right_l[0], right_r[0]):
                stage_channel[right_l[1]][x] = 1

        # Entity channel determines who owns a hitbox or hurtbox
        entity_channel = np.zeros([IMG_SIZE, IMG_SIZE])

        # Create character hurtbox channel.
        # Hurtbox is based on size, since we don't have precise data.
        char_channel = np.zeros([IMG_SIZE, IMG_SIZE])
        for p_id in [1, 2]:
            player: melee.PlayerState = gamestate.players[p_id]
            p_size = self.framedata.characterdata[player.character]["size"] * 2
            pw = int((p_size / 2 / (X_MAX - X_MIN)) * IMG_SIZE)
            ph = int((p_size / (Y_MAX - Y_MIN)) * IMG_SIZE)
            px, py = view_space((player.position.x, player.position.y), IMG_SIZE)
            draw_box(
                px - pw // 2,
                py,
                pw,
                ph,
                char_channel,
            )
            if p_id - 1 == player_id:
                draw_box(
                    px - pw // 2,
                    py,
                    pw,
                    ph,
                    entity_channel,
                )

        # Create hitbox channel.
        hit_channel = np.zeros([IMG_SIZE, IMG_SIZE])

        # Projectiles
        projectiles: list[melee.Projectile] = gamestate.projectiles
        for proj in projectiles:
            px, py = view_space((proj.position.x, proj.position.y), IMG_SIZE)
            if proj.type in PROJECTILE_MAP:
                p_data = PROJECTILE_MAP[proj.type]
                w = int((p_data["w"] / (X_MAX - X_MIN)) * IMG_SIZE)
                h = int((p_data["h"] / (Y_MAX - Y_MIN)) * IMG_SIZE)
                draw_box(px - w // 2, py - h // 2, w, h, hit_channel)

                if proj.owner - 1 == player_id:
                    draw_box(
                        px - w // 2,
                        py - h // 2,
                        w,
                        h,
                        entity_channel,
                    )

        # Attack hitboxes.
        # Hitboxes are based on size, since we don't have precise values.
        for p_id in [1, 2]:
            player = gamestate.players[p_id]
            frame_data = self.framedata.framedata[player.character][player.action][
                player.action_frame
            ]
            if len(frame_data) == 0:
                continue
            for i in range(4):
                hit_id = i + 1
                if frame_data[f"hitbox_{hit_id}_status"]:
                    y = player.position.y + frame_data[f"hitbox_{hit_id}_y"]
                    x = frame_data[f"hitbox_{hit_id}_x"]
                    if not player.facing:
                        x = -x
                    x += player.position.x
                    hx, hy = view_space((x, y), IMG_SIZE)
                    h_size = frame_data[f"hitbox_{hit_id}_size"] * 2
                    w = int((h_size / (X_MAX - X_MIN)) * IMG_SIZE)
                    h = int((h_size / (Y_MAX - Y_MIN)) * IMG_SIZE)
                    draw_box(hx - w // 2, hy - h // 2, w, h, hit_channel)

                    if p_id - 1 == player_id:
                        draw_box(
                            hx - w // 2,
                            hy - h // 2,
                            w,
                            h,
                            entity_channel,
                        )

        return [stage_channel, char_channel, hit_channel, entity_channel]

    def insert_obs(
        self, obs: np.ndarray, frame_stack: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Inserts a new frame and cycles the observation.
        Sets frame "n" to the current value of frame "n - 1", from 1 to `self.num_frames`.
        """
        for i in reversed(range(1, self.num_frames)):
            frame_stack[i] = frame_stack[i - 1]
        frame_stack[0] = obs
        return frame_stack

    def bot_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Non-standard method for single agent envs.
        Returns an observation for the bot. This observation is manually frame
        stacked.
        """
        return (
            np.stack(self.bot_frame_stack),
            np.concatenate([self.bot_stats, self.player_stats]),
        )

    def bot_step(self, action: int):
        """
        Non-standard method for single agent envs.
        Sets the action of the bot. Should be called before `step`.
        """
        self.bot_action = action


def draw_box(x: int, y: int, w: int, h: int, channel: np.ndarray):
    """
    Draws a non rotated box.
    """
    y_max, x_max = channel.shape
    x = max(min(x, x_max - 1), 0)
    y = max(min(y, y_max - 1), 0)
    w = max(max(min(x + w, x_max - 1), 0) - x, 1)
    h = max(max(min(y + h, y_max - 1), 0) - y, 1)
    for y_ in range(y, y + h):
        for x_ in range(x, x + w):
            channel[y_][x_] = 1


def compute_stats_single(
    gamestate: melee.GameState, player_id: int, framedata: melee.FrameData
) -> np.ndarray:
    player: melee.PlayerState = gamestate.players[player_id + 1]
    stats = np.zeros([4 + STATE_COUNT])
    stats[0] = player.percent / 100.0
    stats[1] = int(player.facing)
    stats[2] = player.jumps_left
    stats[3] = player.on_ground
    if player.action in ACTION_MAP:
        stats[4 + ACTION_MAP[player.action]] = 1
    return stats


def view_space(point: tuple[float, float], img_size: int) -> tuple[int, int]:
    """
    Converts a world point into a stage point.
    """
    x, y = point
    x_range = X_MAX - X_MIN
    y_range = Y_MAX - Y_MIN
    x = max(min(int(((x - X_MIN) / x_range) * img_size), img_size - 1), 0)
    y = max(min(int(((y - Y_MIN) / y_range) * img_size), img_size - 1), 0)
    return (x, y)


import os

if __name__ == "__main__":
    env = TimeLimit(
        MeleeEnv(console_id=101, render_mode="human", view_channels=(1, 2, 3)),
        1000,
    )
    act_space = env.action_space
    obs, _ = env.reset()
    while True:
        bot_action = act_space.sample()
        env.env.bot_step(bot_action)  # type: ignore
        action = act_space.sample()
        obs, _, done, trunc, _ = env.step(action)
        env.render()
        if done or trunc:
            obs, _ = env.reset()
