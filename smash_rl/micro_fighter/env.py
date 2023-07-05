"""
Environments for Micro Fighter.
"""
import math
import random
from typing import Any, List, Optional, Tuple
import gymnasium as gym
import numpy as np
from smash_rl_rust import MicroFighter, StepOutput
from PIL import Image, ImageDraw  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import pygame

from smash_rl_rust import GameState

IMG_SIZE = 32
IMG_SCALE = 4


class EnvState:
    """
    This contains internal game state and frame stacked observations.
    """

    def __init__(
        self,
        player_frame_stack: List[np.ndarray],
        bot_frame_stack: List[np.ndarray],
        game_state: GameState,
    ):
        self.player_frame_stack = player_frame_stack
        self.bot_frame_stack = bot_frame_stack
        self.game_state = game_state


class MFEnv(gym.Env):
    """
    An environment that wraps the MicroFighter game.

    ### Observation Space
    4x5 channels of size IMG_SIZE x IMG_SIZE. The channels are:

    0. 1 if hitbox, 0 if hurtbox.
    1. Damage sustained by hurtboxes or inflicted by hitboxes.
    2. 1 if this is the player, 0 if this is the opponent.
    3. State of the character that belongs to the hbox.
    4. 1 if hbox, 0 if empty space.

    We manually perform frame stacking.

    ### Action Space
    0. Do nothing.
    1. Left.
    2. Right.
    3. Jump.
    4. Light.
    5. Heavy.
    6. Shield.
    7. Grab.
    8. Special.
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        view_channels: Tuple[int, int, int] = (0, 1, 2),
        max_skip_frames: int = 0,
        num_frames: int = 4,
    ):
        """
        Args:
        
        render_mode: If "human", renders channels.

        view_channels: Optional 3 string tuple of channels. The channels will be \
            rendered as R, G, and B. See class description for channels.
        
        max_skip_frames: Maximum number of frames that will be skipped. When 0, no \
            frames are skipped. The actual number of frames skipped is random.
        
        num_frames: Number of frames in framestack.
        """

        self.game = MicroFighter(False)
        self.observation_space = gym.spaces.Box(0.0, 1.0, [4, 5, IMG_SIZE, IMG_SIZE])
        self.action_space = gym.spaces.Discrete(9)
        self.render_mode = render_mode
        self.num_frames = num_frames
        self.player_frame_stack = [
            np.zeros([5, IMG_SIZE, IMG_SIZE]) for _ in range(self.num_frames)
        ]
        self.bot_frame_stack = [
            np.zeros([5, IMG_SIZE, IMG_SIZE]) for _ in range(self.num_frames)
        ]
        self.max_skip_frames = max_skip_frames
        self.view_channels = view_channels
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (IMG_SIZE * IMG_SCALE, IMG_SIZE * IMG_SCALE)
            )
            self.clock = pygame.time.Clock()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        skip_frames = random.randrange(0, self.max_skip_frames)
        for _ in range(skip_frames + 1):
            step_output = self.game.step(action)
            if step_output.round_over:
                break

        channels = self.gen_channels(step_output, is_player=True)
        self.insert_obs(np.stack(channels), self.player_frame_stack)
        bot_channels = self.gen_channels(step_output, is_player=False)
        self.insert_obs(np.stack(bot_channels), self.bot_frame_stack)

        terminated = step_output.round_over
        reward = 0.0
        if terminated:
            reward = 1.0 if step_output.player_won else -1.0

        return np.stack(self.player_frame_stack), reward, terminated, False, {}

    def reset(
        self, *args, seed=None, options=None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        step_output = self.game.reset()
        channels = self.gen_channels(step_output, is_player=True)
        self.insert_obs(np.stack(channels), self.player_frame_stack)
        bot_channels = self.gen_channels(step_output, is_player=False)
        self.insert_obs(np.stack(bot_channels), self.bot_frame_stack)
        return np.stack(self.player_frame_stack), {}

    def gen_channels(
        self, step_output: StepOutput, is_player: bool = False
    ) -> List[np.ndarray]:
        """
        Converts `step_output` into observation channels.
        If `is_player` is not true, this will be for the bot.
        """
        hboxes = step_output.hboxes
        hit_channel = np.zeros([IMG_SIZE, IMG_SIZE])
        dmg_channel = np.zeros([IMG_SIZE, IMG_SIZE])
        player_channel = np.zeros([IMG_SIZE, IMG_SIZE])
        state_channel = np.zeros([IMG_SIZE, IMG_SIZE])
        box_channel = np.zeros([IMG_SIZE, IMG_SIZE])
        for hbox in hboxes:
            box_img = Image.new("1", (IMG_SIZE, IMG_SIZE))
            box_draw = ImageDraw.ImageDraw(box_img)
            rot = -hbox.angle
            # Rotate points around center
            points = np.array(
                [
                    (-hbox.w / 2, hbox.h / 2),
                    (hbox.w / 2, hbox.h / 2),
                    (hbox.w / 2, -hbox.h / 2),
                    (-hbox.w / 2, -hbox.h / 2),
                ]
            ) @ np.array(
                [[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]]
            )
            # Offset points by x and y
            points = points + np.array([[hbox.x + hbox.w / 2, hbox.y - hbox.y / 2]])
            # Convert to image space
            points = (points + self.game.get_screen_size() / 2) * (
                IMG_SIZE / self.game.get_screen_size()
            )
            # Convert to integer coordinates
            points = points.astype(np.int32)
            points_list = [
                list(points[0]),
                list(points[1]),
                list(points[2]),
                list(points[3]),
            ]
            box_draw.polygon([(p[0], p[1]) for p in points_list], fill=1)
            box_arr = np.array(box_img)
            inv_box_arr = 1 - box_arr
            hit_channel = hit_channel * inv_box_arr + box_arr * float(hbox.is_hit)
            dmg_channel = dmg_channel * inv_box_arr + box_arr * (hbox.damage / 100)
            player_channel = player_channel * inv_box_arr + box_arr * float(
                hbox.is_player == is_player
            )
            state_channel = state_channel * inv_box_arr + box_arr * (
                int(hbox.move_state) / 8
            )
            if hbox.is_player:
                print(f"Player state: {hbox.move_state}")
            else:
                print(f"Bot state: {hbox.move_state}")
            box_channel = box_channel * inv_box_arr + box_arr
        return [
            hit_channel,
            dmg_channel,
            player_channel,
            state_channel,
            box_channel,
        ]

    def insert_obs(self, obs: np.ndarray, frame_stack: list[np.ndarray]):
        """
        Inserts a new frame and cycles the observation.
        Sets frame "n" to the current value of frame "n - 1", from 1 to `self.num_frames`.
        """
        for i in reversed(range(1, self.num_frames)):
            frame_stack[i] = frame_stack[i - 1]
        frame_stack[0] = obs

    def render(self):
        if self.render_mode == "human":
            channels = self.player_frame_stack[0]
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
            self.clock.tick(60)

    def player_obs(self) -> np.ndarray:
        """
        Non-standard method for single agent envs.
        Returns an observation for the player. This observation is manually frame
        stacked.
        """
        return np.stack(self.player_frame_stack)

    def bot_obs(self) -> np.ndarray:
        """
        Non-standard method for single agent envs.
        Returns an observation for the bot. This observation is manually frame
        stacked.
        """
        return np.stack(self.bot_frame_stack)

    def bot_step(self, action: int):
        """
        Non-standard method for single agent envs.
        Sets the action of the bot. Should be called before `step`.
        """
        self.game.bot_step(action)

    def state(self) -> EnvState:
        """
        Returns the current state of the environment.
        """
        return EnvState(self.player_frame_stack, self.bot_frame_stack, self.game.get_game_state())

    def load_state(self, state: EnvState):
        """
        Loads a previously saved state.
        """
        self.player_frame_stack = state.player_frame_stack
        self.bot_frame_stack = state.bot_frame_stack
        self.game.load_state(state.game_state)
