"""
Environments for Micro Fighter.
"""
import math
import random
from typing import Any, Callable, List, Optional, Tuple
import gymnasium as gym
import numpy as np
from smash_rl_rust import MicroFighter, StepOutput
from PIL import Image, ImageDraw  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import pygame
from horapy import HNSWIndex  # type: ignore
import json
from smash_rl_rust import GameState
from pathlib import Path
from sklearn.decomposition import PCA  # type: ignore
from torch import nn
import torch

IMG_SIZE = 32
IMG_SCALE = 8


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


class BaseMFEnv(gym.Env):
    """
    Base class for MicroFighter environments.
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        view_channels: Tuple[int, int, int] = (0, 1, 2),
        max_skip_frames: int = 0,
        num_frames: int = 4,
    ):
        self.game = MicroFighter(False)
        self.num_channels = 4
        self.observation_space = gym.spaces.Tuple(
            [
                gym.spaces.Box(
                    -1.0, 1.0, [num_frames, self.num_channels, IMG_SIZE, IMG_SIZE]
                ),
                gym.spaces.Box(-1.0, 2.0, [36]),
            ]
        )
        self.action_space = gym.spaces.Discrete(9)
        self.player_stats = np.zeros([18])
        self.bot_stats = np.zeros([18])
        self.bot_action = 0
        self.render_mode = render_mode
        self.num_frames = num_frames
        self.player_frame_stack = [
            np.zeros([self.num_channels, IMG_SIZE, IMG_SIZE])
            for _ in range(self.num_frames)
        ]
        self.bot_frame_stack = [
            np.zeros([self.num_channels, IMG_SIZE, IMG_SIZE])
            for _ in range(self.num_frames)
        ]
        self.max_skip_frames = max_skip_frames
        self.view_channels = view_channels
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (IMG_SIZE * IMG_SCALE, IMG_SIZE * IMG_SCALE)
            )
            self.clock = pygame.time.Clock()

    def gen_channels(
        self, step_output: StepOutput, is_player: bool
    ) -> List[np.ndarray]:
        """
        Converts `step_output` into observation channels.
        If `is_player` is not true, this will be for the bot.
        """
        hboxes = step_output.hboxes
        hit_channel = np.zeros([IMG_SIZE, IMG_SIZE])
        dmg_channel = np.zeros([IMG_SIZE, IMG_SIZE])
        player_channel = np.zeros([IMG_SIZE, IMG_SIZE])
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
            points = points + np.array([[hbox.x + hbox.w / 2, hbox.y + hbox.h / 2]])
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
            dmg_channel = dmg_channel * inv_box_arr + box_arr * (hbox.damage / 10)
            player_channel = player_channel * inv_box_arr + box_arr * float(
                hbox.is_player == is_player
            )
            box_channel = box_channel * inv_box_arr + box_arr
        return [
            hit_channel,
            dmg_channel,
            player_channel,
            box_channel,
        ]

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

    def compute_stats(self, step_output: StepOutput):
        self.player_stats = np.zeros([18])
        self.player_stats[int(step_output.player_state)] = 1
        self.player_stats[16] = step_output.player_damage / 100.0
        self.player_stats[17] = step_output.player_dir
        self.bot_stats = np.zeros([18])
        self.bot_stats[int(step_output.opponent_state)] = 1
        self.bot_stats[16] = step_output.opponent_damage / 100.0
        self.bot_stats[17] = step_output.opponent_dir

    def base_reset(
        self,
    ) -> tuple[tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]], StepOutput]:
        step_output = self.game.reset()
        self.player_frame_stack = [
            np.zeros([self.num_channels, IMG_SIZE, IMG_SIZE])
            for _ in range(self.num_frames)
        ]
        self.bot_frame_stack = [
            np.zeros([self.num_channels, IMG_SIZE, IMG_SIZE])
            for _ in range(self.num_frames)
        ]
        channels = self.gen_channels(step_output, is_player=True)
        self.player_frame_stack = self.insert_obs(
            np.stack(channels), self.player_frame_stack
        )
        bot_channels = self.gen_channels(step_output, is_player=False)
        self.bot_frame_stack = self.insert_obs(
            np.stack(bot_channels), self.bot_frame_stack
        )

        self.compute_stats(step_output)
        stats_obs = np.concatenate([self.player_stats, self.bot_stats])

        return ((np.stack(self.player_frame_stack), stats_obs), {}), step_output

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
            self.clock.tick(60 / self.num_frames)

    def player_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Non-standard method for single agent envs.
        Returns an observation for the player. This observation is manually frame
        stacked.
        """
        return (
            np.stack(self.player_frame_stack),
            np.concatenate([self.player_stats, self.bot_stats]),
        )

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

    def state(self) -> EnvState:
        """
        Returns the current state of the environment.
        """
        return EnvState(
            self.player_frame_stack, self.bot_frame_stack, self.game.get_game_state()
        )

    def load_state(self, state: EnvState):
        """
        Loads a previously saved state.
        """
        self.player_frame_stack = state.player_frame_stack
        self.bot_frame_stack = state.bot_frame_stack
        self.game.load_state(state.game_state)


class MFEnv(BaseMFEnv):
    """
    An environment that wraps the MicroFighter game.

    ### Observation Space

    #### Spatial:
    4x4 channels of size IMG_SIZE x IMG_SIZE. The channels are:

    0. 1 if hitbox, 0 if hurtbox.
    1. Damage inflicted by hitboxes.
    2. 1 if this is the player, 0 if this is the opponent.
    3. 1 if hbox, 0 if empty space.

    We manually perform frame stacking.

    #### Stats:
    0 - 15. Player state.
    16. Player damage.
    17. Direction, -1 if left, 1 if right.
    18 - 33. Opponent state.
    34. Opponent damage.
    35. Direction, -1 if left, 1 if right.

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
            frames are skipped. Right now, this is deterministic.
        
        num_frames: Number of frames in framestack.
        """
        super().__init__(render_mode, view_channels, max_skip_frames, num_frames)
        self.dmg_reward_amount = 1.0
        self.last_dist = 0.0

    def step(
        self, action: int
    ) -> tuple[tuple[np.ndarray, np.ndarray], float, bool, bool, dict[str, Any]]:
        skip_frames = self.max_skip_frames
        dmg_reward = 0.0
        for _ in range(skip_frames + 1):
            self.game.bot_step(self.bot_action)
            step_output = self.game.step(action)
            dmg_reward += step_output.net_damage
            if step_output.round_over:
                break

        # Spatial observation
        channels = self.gen_channels(step_output, is_player=True)
        self.player_frame_stack = self.insert_obs(
            np.stack(channels), self.player_frame_stack
        )
        bot_channels = self.gen_channels(step_output, is_player=False)
        self.bot_frame_stack = self.insert_obs(
            np.stack(bot_channels), self.bot_frame_stack
        )

        # Stats observation
        self.compute_stats(step_output)
        stats_obs = np.concatenate([self.player_stats, self.bot_stats])

        terminated = step_output.round_over
        round_reward = 0.0
        if terminated:
            round_reward = 1.0 if step_output.player_won else -1.0

        dmg_reward = dmg_reward / 10

        reward = dmg_reward * (self.dmg_reward_amount) + round_reward

        return (
            (np.stack(self.player_frame_stack), stats_obs),
            reward,
            terminated,
            False,
            {"player_won": step_output.player_won},
        )

    def reset(
        self, *args, seed=None, options=None
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
        reset_data, step_output = self.base_reset()

        self.last_dist = step_output.player_pos[0] ** 2 / 200**2

        return reset_data

    def set_dmg_reward_amount(self, amount: float):
        """
        Sets how much the damage reward will be added to the overall reward.
        """
        self.dmg_reward_amount = amount


class CurriculumEnv(BaseMFEnv):
    """
    Environment used in curriculum learning setting.
    """

    def __init__(
        self,
        reward_fn: Callable[[StepOutput], tuple[float, bool]],
        bot_actions_fn: Callable[[], int],
        render_mode: Optional[str] = None,
        view_channels: Tuple[int, int, int] = (0, 1, 2),
        max_skip_frames: int = 0,
        num_frames: int = 4,
    ):
        """
        reward_fn: Function that accepts a StepOutput and returns the reward and terminated flag.
        """
        super().__init__(render_mode, view_channels, max_skip_frames, num_frames)
        self.reward_fn = reward_fn
        self.bot_actions_fn = bot_actions_fn

    def step(
        self, action: int
    ) -> tuple[tuple[np.ndarray, np.ndarray], float, bool, bool, dict[str, Any]]:
        bot_action = self.bot_actions_fn()
        self.game.bot_step(bot_action)

        skip_frames = self.max_skip_frames
        for _ in range(skip_frames + 1):
            step_output = self.game.step(action)
            if step_output.round_over:
                break

        # Spatial observation
        channels = self.gen_channels(step_output, is_player=True)
        self.player_frame_stack = self.insert_obs(
            np.stack(channels), self.player_frame_stack
        )
        bot_channels = self.gen_channels(step_output, is_player=False)
        self.bot_frame_stack = self.insert_obs(
            np.stack(bot_channels), self.bot_frame_stack
        )

        # Stats observation
        self.compute_stats(step_output)
        stats_obs = np.concatenate([self.player_stats, self.bot_stats])

        reward, terminated = self.reward_fn(step_output)

        return (
            (np.stack(self.player_frame_stack), stats_obs),
            reward,
            terminated,
            False,
            {"player_won": step_output.player_won},
        )

    def reset(
        self, *args, seed=None, options=None
    ) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
        reset_data, step_output = self.base_reset()
        return reset_data


class RetrievalContext:
    """
    Central location for data needed for retrieval.
    """

    def __init__(
        self,
        key_dim: int,
        index_path: str,
        generated_dir: str,
        episode_data_path: str,
        encoder: nn.Module,
        pca: PCA,
    ):
        self.encoder = encoder
        self.pca = pca
        print("Loading retrieval context...", end="")
        # Load index
        self.index = HNSWIndex(key_dim, "usize")
        self.index.load(index_path)
        self.index.build("dot_product")

        # Set up episode data
        traj_data = []
        with open(episode_data_path, "r") as f:
            episode_data = json.load(f)
        for data in episode_data:
            traj_in_episode = len(data.pop("traj_in_episode"))
            for _ in range(traj_in_episode):
                traj_data.append(data)

        # Load trajectory data
        data_path = Path(generated_dir)
        file_count = len([item for item in data_path.iterdir()]) // 2
        data_spatial = []
        data_scalar = []
        for i in range(file_count):
            data_spatial.append(np.load(data_path / f"{i}_data_spatial.npy"))
            data_scalar.append(np.load(data_path / f"{i}_data_scalar.npy"))
        self.data_spatial = np.concatenate(data_spatial, 0)
        self.data_scalar = np.concatenate(data_scalar, 0)
        del data_spatial, data_scalar
        extra_scalar = np.zeros([self.data_scalar.shape[0], 1])
        for i in range(len(traj_data)):
            player_won = traj_data[i]["player_won"]
            extra_scalar[i][0] = player_won
        self.data_scalar = np.concatenate([self.data_scalar, extra_scalar], 1)
        print("Done.")

    def search(self, key: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns top k spatial and scalar data.
        First dimension of both is `top_k`.
        """
        indices = np.array(self.index.search(key, top_k))
        results_spatial = self.data_spatial[indices]
        results_scalar = self.data_scalar[indices]
        return (results_spatial, results_scalar)

    def search_from_obs(
        self, spatial: np.ndarray, stats: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns top k spatial and scalar data.
        First dimension of both is `top_k`.
        """
        with torch.no_grad():
            encoded = self.encoder(
                torch.from_numpy(spatial).unsqueeze(0).float(),
                torch.from_numpy(stats).unsqueeze(0).float(),
            ).numpy()
            key = self.pca.transform(encoded).squeeze()
        return self.search(key, top_k)


class RetrievalMFEnv(MFEnv):
    def __init__(
        self,
        retrieval_ctx: RetrievalContext,
        top_k: int,
        render_mode: Optional[str] = None,
        view_channels: Tuple[int, int, int] = (0, 1, 2),
        max_skip_frames: int = 0,
        num_frames: int = 4,
    ):
        """
        A retrieval augmented environment, where each observation is augmented
        with top K metadata items from a large dataset.
        Other than that, this is the exact same as MFEnv.
        """
        super().__init__(render_mode, view_channels, max_skip_frames, num_frames)
        self.retrieval_ctx = retrieval_ctx
        self.top_k = top_k
        self.observation_space = gym.spaces.Tuple(
            [
                # Standard observation
                gym.spaces.Box(
                    -1.0, 1.0, [num_frames, self.num_channels, IMG_SIZE, IMG_SIZE]
                ),
                gym.spaces.Box(-1.0, 2.0, [36]),
                # Neighbor observations
                gym.spaces.Box(-1.0, 1.0, [top_k, 16, IMG_SIZE, IMG_SIZE]),
                gym.spaces.Box(-1.0, 2.0, [top_k, 181]),
            ]
        )

    def step(  # type: ignore
        self, action: int
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float, bool, bool, dict[str, Any]]:  # type: ignore
        (spatial, stats), reward, done, trunc, info = super().step(action)
        neighbor_spatial, neighbor_scalar = self.retrieval_ctx.search_from_obs(
            spatial, stats, self.top_k
        )
        return (
            (spatial, stats, neighbor_spatial, neighbor_scalar),
            reward,
            done,
            trunc,
            info,
        )

    def reset(  # type: ignore
        self, *args, seed=None, options=None
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
        ((spatial, stats), info), step_output = self.base_reset()
        neighbor_spatial, neighbor_scalar = self.retrieval_ctx.search_from_obs(
            spatial, stats, self.top_k
        )
        return (spatial, stats, neighbor_spatial, neighbor_scalar), info

    def bot_obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # type: ignore
        spatial, stats = super().bot_obs()
        neighbor_spatial, neighbor_scalar = self.retrieval_ctx.search_from_obs(
            spatial, stats, self.top_k
        )
        return (spatial, stats, neighbor_spatial, neighbor_scalar)
