"""
Trains an agent with PPO.
This experiment uses a curriculum to teach the agent certain skills.
"""
from argparse import ArgumentParser
import copy
from functools import reduce
import random
import time
from typing import Any, Mapping, Union
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import numpy as np

import torch
import torch.nn as nn
import wandb
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.normalize import NormalizeReward
from torch.distributions import Categorical
from tqdm import tqdm

from smash_rl.algorithms.ppo_multi import train_ppo
from smash_rl.algorithms.rollout_buffer import RolloutBuffer, StateRolloutBuffer
from smash_rl.conf import entity
from smash_rl.micro_fighter.env import CurriculumEnv as MFEnv
from smash_rl.utils import init_orthogonal
from smash_rl_rust import StepOutput, test_jit

_: Any

# Hyperparameters
num_envs = 16  # Number of environments to step through at once during sampling.
train_steps = 256  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs.
iterations = 1000  # Number of sample/train iterations.
train_iters = 1  # Number of passes over the samples collected.
train_batch_size = 512  # Minibatch size while training models.
discount = 0.995  # Discount factor applied to rewards.
lambda_ = 0.95  # Lambda for GAE.
epsilon = 0.2  # Epsilon for importance sample clipping.
v_lr = 0.001  # Learning rate of the value net.
p_lr = 0.0003  # Learning rate of the policy net.
num_frames = 4  # Number of frames in frame stack.
max_skip_frames = 1  # Max number of frames to skip.
time_limit = 500  # Time limit before truncation.
elo_k = 16  # ELO adjustment constant.
eval_every = 2  # Number of iterations before evaluating.
eval_steps = 5  # Number of eval runs to perform.
max_eval_steps = 500  # Max number of steps to take during each eval run.
device = torch.device("cuda")  # Device to use during training.

# Argument parsing
parser = ArgumentParser()
parser.add_argument("--eval", action="store_true")
parser.add_argument("--generate", action="store", default=None)
parser.add_argument("--trace", action="store_true")
args = parser.parse_args()


class SharedNet(nn.Module):
    """
    A shared architecture for the value and policy nets.
    """

    def __init__(self, obs_shape_spatial: torch.Size, obs_shape_stats: int):
        nn.Module.__init__(self)
        channels = obs_shape_spatial[0] * obs_shape_spatial[1]  # Frames times channels
        self.spatial_net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 256, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2),
            nn.ReLU(),
        )
        self.stats_net = nn.Sequential(
            nn.Linear(obs_shape_stats, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

    def forward(self, spatial: torch.Tensor, stats: torch.Tensor):
        spatial = torch.flatten(spatial, 1, 2)
        spatial = self.spatial_net(spatial)
        spatial = torch.max(torch.max(spatial, dim=3).values, dim=2).values

        stats = self.stats_net(stats)

        x = torch.concat([spatial, stats], dim=1)
        x = self.net2(x)
        return x


class ValueNet(nn.Module):
    def __init__(self, obs_shape_spatial: torch.Size, obs_shape_stats: int):
        nn.Module.__init__(self)
        self.shared = SharedNet(obs_shape_spatial, obs_shape_stats)
        self.net = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        init_orthogonal(self)

    def forward(self, spatial: torch.Tensor, stats: torch.Tensor):
        x = self.shared(spatial, stats)
        x = self.net(x)
        return x


class PolicyNet(nn.Module):
    def __init__(
        self, obs_shape_spatial: torch.Size, obs_shape_stats: int, action_count: int
    ):
        nn.Module.__init__(self)
        self.shared = SharedNet(obs_shape_spatial, obs_shape_stats)
        self.net = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_count),
            nn.LogSoftmax(1),
        )
        init_orthogonal(self)

    def forward(self, spatial: torch.Tensor, stats: torch.Tensor):
        x = self.shared(spatial, stats)
        x = self.net(x)
        return x


def simple_reward(step_output: StepOutput) -> tuple[float, bool]:
    reward = -1
    done = step_output.round_over
    if done:
        if step_output.player_won:
            reward = 100
        else:
            reward = -100
    return reward, done

def attack_actions() -> int:
    return 0

def block_actions() -> int:
    return 8

def grab_actions() -> int:
    return 6

env = SyncVectorEnv(
    [
        lambda: NormalizeReward(
            TimeLimit(
                MFEnv(
                    simple_reward,
                    attack_actions,
                    max_skip_frames=max_skip_frames,
                    num_frames=num_frames,
                ),
                time_limit,
            )
        )
        for _ in range(num_envs)
    ]
)
test_env = TimeLimit(
    MFEnv(simple_reward,
                    attack_actions, max_skip_frames=max_skip_frames, num_frames=num_frames),
    max_eval_steps,
)

# If evaluating, load the latest policy
if args.eval:
    eval_done = False
    obs_space = env.single_observation_space
    act_space = env.single_action_space
    assert isinstance(obs_space, gym.spaces.Tuple)
    assert isinstance(obs_space.spaces[0], gym.spaces.Box)
    assert isinstance(obs_space.spaces[1], gym.spaces.Box)
    spatial_obs_space = obs_space.spaces[0]
    stats_obs_space = obs_space.spaces[1]
    assert isinstance(act_space, gym.spaces.Discrete)
    p_net = PolicyNet(
        torch.Size(spatial_obs_space.shape), stats_obs_space.shape[0], int(act_space.n)
    )
    p_net.load_state_dict(torch.load("temp/p_net.pt"))
    test_env = TimeLimit(
        MFEnv(
            simple_reward,
                    attack_actions,
            max_skip_frames=max_skip_frames,
            render_mode="human",
            view_channels=(0, 2, 3),
            num_frames=num_frames,
        ),
        max_eval_steps,
    )
    with torch.no_grad():
        (obs_1_, obs_2_), _ = test_env.reset()
        eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
        eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
        while True:
            action_probs = p_net(
                eval_obs_1.unsqueeze(0), eval_obs_2.unsqueeze(0)
            ).squeeze()
            action = Categorical(logits=action_probs).sample().numpy()
            (
                (obs_1_, obs_2_),
                reward,
                eval_done,
                eval_trunc,
                eval_info,
            ) = test_env.step(action)

            test_env.render()
            eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
            eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
            if eval_done or eval_trunc:
                (obs_1_, obs_2_), _ = test_env.reset()
                eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()

wandb.init(
    project="smash-rl",
    entity=entity,
    config={
        "experiment": "micro fighter ppo with curriculum",
        "num_envs": num_envs,
        "train_steps": train_steps,
        "train_iters": train_iters,
        "train_batch_size": train_batch_size,
        "discount": discount,
        "lambda": lambda_,
        "epsilon": epsilon,
        "v_lr": v_lr,
        "p_lr": p_lr,
    },
)

# Initialize policy and value networks
obs_space = env.single_observation_space
act_space = env.single_action_space
assert isinstance(obs_space, gym.spaces.Tuple)
assert isinstance(obs_space.spaces[0], gym.spaces.Box)
assert isinstance(obs_space.spaces[1], gym.spaces.Box)
spatial_obs_space = obs_space.spaces[0]
stats_obs_space = obs_space.spaces[1]
assert isinstance(act_space, gym.spaces.Discrete)
assert isinstance(env.envs[0], NormalizeReward)
assert isinstance(env.envs[0].env, TimeLimit)
assert isinstance(env.envs[0].env.env, MFEnv)
v_net = ValueNet(torch.Size(spatial_obs_space.shape), stats_obs_space.shape[0])
p_net = PolicyNet(
    torch.Size(spatial_obs_space.shape), stats_obs_space.shape[0], int(act_space.n)
)
v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)

buffer_spatial = RolloutBuffer(
    torch.Size(spatial_obs_space.shape),
    torch.Size((1,)),
    torch.Size((int(act_space.n),)),
    torch.int,
    num_envs,
    train_steps,
)
buffer_stats = StateRolloutBuffer(
    torch.Size(stats_obs_space.shape), num_envs, train_steps
)

(obs_1_, obs_2_), _ = env.reset()
obs_1 = torch.from_numpy(obs_1_).float()
obs_2 = torch.from_numpy(obs_2_).float()
for step in tqdm(range(iterations), position=0):
    percent_done = step / iterations

    # Collect experience
    with torch.no_grad():
        total_entropy = 0.0
        for _ in tqdm(range(train_steps), position=1):
            # Choose player action
            action_probs = p_net(obs_1, obs_2)
            action_distr = Categorical(logits=action_probs)
            total_entropy += action_distr.entropy().mean()
            actions = action_distr.sample().numpy()
            (obs_spatial_, obs_stats_), rewards, dones, truncs, _ = env.step(actions)

            try:
                (obs_1_, obs_2_), rewards, dones, truncs, _ = env.step(actions)
                buffer_spatial.insert_step(
                    obs_1,
                    torch.from_numpy(actions).unsqueeze(1),
                    action_probs,
                    list(rewards),
                    list(dones),
                    list(truncs),
                    None,
                )
                buffer_stats.insert_step(obs_2)
                obs_1 = torch.from_numpy(obs_1_).float()
                obs_2 = torch.from_numpy(obs_2_).float()

            except KeyboardInterrupt:
                quit()
            except Exception as e:
                print(f"Exception in simulation: {e}. Resetting.")
                (obs_1_, obs_2_), _ = env.reset()
                obs_1 = torch.from_numpy(obs_1_).float()
                obs_2 = torch.from_numpy(obs_2_).float()
        buffer_spatial.insert_final_step(obs_1)
        buffer_stats.insert_final_step(obs_2)

    # Train
    total_p_loss, total_v_loss = train_ppo(
        p_net,
        v_net,
        p_opt,
        v_opt,
        buffer_spatial,
        [buffer_stats],
        device,
        train_iters,
        train_batch_size,
        discount,
        lambda_,
        epsilon,
        entropy_coeff=0.0,
    )
    buffer_spatial.clear()
    buffer_stats.clear()

    # Evaluate the network's performance.
    if (step + 1) % eval_every == 0:
        eval_done = False
        total_reward = 0.0
        with torch.no_grad():
            (obs_1_, obs_2_), _ = test_env.reset()
            eval_obs_1 = torch.from_numpy(obs_1_).float()
            eval_obs_2 = torch.from_numpy(obs_2_).float()
            for _ in range(eval_steps):
                try:
                    for _ in range(max_eval_steps):
                        action_probs = p_net(
                            eval_obs_1.unsqueeze(0), eval_obs_2.unsqueeze(0)
                        ).squeeze()
                        actions = Categorical(logits=action_probs).sample().numpy()
                        (
                            (obs_1_, obs_2_),
                            reward,
                            eval_done,
                            eval_trunc,
                            eval_info,
                        ) = test_env.step(actions)

                        total_reward += reward

                        eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                        eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
                        if eval_done or eval_trunc:
                            (obs_1_, obs_2_), info = test_env.reset()
                            eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                            eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
                            break
                except KeyboardInterrupt:
                    quit()
                except Exception as e:
                    print(f"Exception during evaluation: {e}.")
                    pass
        wandb.log(
            {
                "total_eval_reward": total_reward,
            },
        )

    wandb.log(
        {
            "avg_v_loss": total_v_loss / train_iters,
            "avg_p_loss": total_p_loss / train_iters,
            "entropy": total_entropy / train_steps,
        }
    )

    torch.save(p_net.state_dict(), "temp/p_net.pt")
    torch.save(v_net.state_dict(), "temp/v_net.pt")
