"""
Environments for Micro Fighter.
"""
import math
from typing import Any, List, Optional, Tuple
import gymnasium as gym
import numpy as np
from smash_rl_rust import MicroFighter, StepOutput
from PIL import Image, ImageDraw  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import pygame

IMG_SIZE = 32
IMG_SCALE = 4


class MFEnv(gym.Env):
    """
    An environment that wraps the MicroFighter game.

    ### Observation Space
    4 channels of size IMG_SIZE x IMG_SIZE. The channels are:

    0. 1 if hitbox, 0 if hurtbox.
    1. Damage sustained by hurtboxes or inflicted by hitboxes.
    2. 1 if this is the player, 0 if this is the opponent.
    3. State of the character that belongs to the hbox.
    4. 1 if hbox, 0 if empty space.

    ### Action Space
    0. Do nothing.
    1. Left.
    2. Right.
    3. Jump.
    4. Light.
    5. Heavy.
    6. Shield.
    7. Grab.
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        view_channels: Optional[Tuple[int, int, int]] = None,
    ):
        """
        Args:
        
        render_mode: If "human", renders channels.

        view_channels: Optinal 3 string tuple of channels. The channels will be \
            rendered as R, G, and B. See class description for channels.
        """

        self.game = MicroFighter(False)
        self.observation_space = gym.spaces.Box(0.0, 1.0, [5, IMG_SIZE, IMG_SIZE])
        self.action_space = gym.spaces.Discrete(8)
        self.render_mode = render_mode
        self.channels = [np.zeros([IMG_SIZE, IMG_SIZE]) for _ in range(5)]
        self.view_channels = (0, 1, 2)
        if view_channels:
            self.view_channels = view_channels
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (IMG_SIZE * IMG_SCALE, IMG_SIZE * IMG_SCALE)
            )
            self.clock = pygame.time.Clock()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        step_output = self.game.step(action)
        self.channels = self.gen_channels(step_output)

        terminated = step_output.round_over
        reward = 0.0
        if terminated:
            reward = 1.0 if step_output.player_won else -1.0

        return np.stack(self.channels), reward, False, terminated, {}

    def reset(
        self, *args, seed=None, options=None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        step_output = self.game.reset()
        self.channels = self.gen_channels(step_output)
        return np.stack(self.channels), {}

    def gen_channels(self, step_output: StepOutput) -> List[np.ndarray]:
        """
        Converts `step_output` into observation channels.
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
                hbox.is_player
            )
            state_channel = state_channel * inv_box_arr + box_arr * (
                int(hbox.char_state) / 8
            )
            box_channel = box_channel * inv_box_arr + box_arr
        return [
            hit_channel,
            dmg_channel,
            player_channel,
            state_channel,
            box_channel,
        ]

    def render(self):
        if self.render_mode == "human":
            r = self.channels[self.view_channels[0]]
            g = self.channels[self.view_channels[1]]
            b = self.channels[self.view_channels[2]]
            view = np.flip(np.stack([r, g, b]).transpose(2, 1, 0), 1).clip(0, 1) * 255.0
            view_surf = pygame.Surface([IMG_SIZE, IMG_SIZE])
            pygame.surfarray.blit_array(view_surf, view)
            pygame.transform.scale(
                view_surf, [IMG_SIZE * IMG_SCALE, IMG_SIZE * IMG_SCALE], self.screen
            )
            pygame.display.flip()
            self.clock.tick(60)
