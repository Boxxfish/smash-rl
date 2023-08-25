import time
from typing import Any, Optional, SupportsFloat
from gymnasium import Env
import gymnasium as gym
import numpy as np
import melee  # type: ignore
from melee import Button, Action
import random
from gymnasium.wrappers.time_limit import TimeLimit

IMG_SIZE = 64

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
    # Throw
    Action.THROW_BACK: 26,
    Action.THROW_DOWN: 26,
    Action.THROW_FORWARD: 26,
    Action.THROW_UP: 26,
}
STATE_COUNT = max(ACTION_MAP.values()) + 1


class MeleeEnv(Env):
    """
    Environment that wraps a Dolphin emulator.
    The agent is randomly assigned as player 1 or 2.
    The action space is [buttons (5), main stick (7), control stick (5)].
    """

    def __init__(self, dolphin_home_path: str):
        self.console = melee.Console(
            dolphin_home_path=dolphin_home_path, tmp_home_directory=False
        )
        self.console.connect()
        self.action_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(5), gym.spaces.Discrete(7), gym.spaces.Discrete(5)]
        )
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
        self.last_button: Optional[int] = None
        self.framedata = melee.framedata.FrameData()
        self.last_stocks1 = 0
        self.last_stocks2 = 0

    def step(
        self, action: tuple[int, int, int]
    ) -> tuple[tuple[np.ndarray, np.ndarray], float, bool, bool, dict[str, Any]]:
        # Handle inputs
        button, main_stick, c_stick = action
        if self.last_button is not button:
            self.pads[self.player_pad].press_button(self.button_list[button])
            if self.last_button is not None:
                self.pads[self.player_pad].release_button(
                    self.button_list[self.last_button]
                )
        self.last_button = button
        main_stick_dir = self.main_stick_dir[main_stick]
        c_stick_dir = self.c_stick_dir[c_stick]
        self.pads[self.player_pad].tilt_analog(
            Button.BUTTON_MAIN, main_stick_dir[0], main_stick_dir[1]
        )
        self.pads[self.player_pad].tilt_analog(
            Button.BUTTON_C, c_stick_dir[0], c_stick_dir[1]
        )

        # Step and get gamestate
        gamestate = self.console.step()

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
                reward = -1.0
            else:
                reward = 1.0
        
        if diff_stocks2 == -1:
            if self.player_pad == 1:
                reward = -1.0
            else:
                reward = 1.0

        if stocks1 == 0 or stocks2 == 0:
            done = True

        print(reward)

        # Generate stats
        self.player_stats = compute_stats_single(gamestate, self.player_pad, self.framedata)
        self.bot_stats = compute_stats_single(gamestate, 1 - self.player_pad, self.framedata)

        return (np.array([]), np.array([])), reward, done, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
        # Reload state
        self.pads[0].press_button(Button.BUTTON_D_DOWN)
        self.pads[0].flush()
        time.sleep(0.1)
        self.pads[0].release_all()
        self.pads[0].flush()
        self.pads[1].flush()

        # Hold down speed delimiter
        # self.pads[0].press_button(Button.BUTTON_D_RIGHT)
        # self.pads[0].flush()

        # Two second delay to skip loading screen
        time.sleep(2.0)

        self.player_pad = random.randrange(0, 2)

        # Generate stats
        gamestate = self.console.step()
        self.player_stats = compute_stats_single(gamestate, self.player_pad, self.framedata)
        self.bot_stats = compute_stats_single(gamestate, 1 - self.player_pad, self.framedata)
        self.last_stocks1 = int(gamestate.players[1].stock)
        self.last_stocks2 = int(gamestate.players[2].stock)

        return (np.array([]), np.array([])), {}


def compute_stats_single(
    gamestate: melee.GameState, player_id: int, framedata: melee.FrameData
) -> np.ndarray:
    player: melee.PlayerState = gamestate.players[player_id + 1]
    stats = np.zeros([3 + STATE_COUNT])
    stats[0] = player.percent / 100.0
    stats[1] = int(player.facing)
    stats[2] = player.jumps_left
    if player.action in ACTION_MAP:
        stats[3 + ACTION_MAP[player.action]] = player.action_frame / framedata.frame_count(
            player.character, player.action
        )
    return stats


if __name__ == "__main__":
    env = TimeLimit(MeleeEnv("/home/ben/.config/SlippiOnline/"), 1000)
    act_space = env.action_space
    obs, _ = env.reset()
    while True:
        action = act_space.sample()
        obs, _, done, trunc, _ = env.step(action)
        if done or trunc:
            obs, _ = env.reset()
