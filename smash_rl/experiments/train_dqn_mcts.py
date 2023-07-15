"""
Trains an agent with a DQN combined with MCTS.
"""
import copy
import random
from typing import Any

import numpy as np  # type: ignore
import torch
import torch.nn as nn
import wandb
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers.time_limit import TimeLimit
import gymnasium as gym
from tqdm import tqdm
from argparse import ArgumentParser

import smash_rl.conf
from smash_rl.algorithms.dqn import train_dqn
from smash_rl.algorithms.replay_buffer import ReplayBuffer
from smash_rl.micro_fighter.env import MFEnv
from smash_rl.utils import init_orthogonal
from smash_rl.mcts import run_mcts

_: Any
INF = 10**8

# Hyperparameters
num_envs = 1  # Number of environments to step through at once during sampling.
train_steps = 128  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs.
iterations = 100000  # Number of sample/train iterations.
train_iters = 1  # Number of passes over the samples collected.
train_batch_size = 64  # Minibatch size while training models.
discount = 0.99  # Discount factor applied to rewards.
q_epsilon = 0.9  # Epsilon for epsilon greedy strategy. This gets annealed over time.
eval_steps = 1  # Number of eval runs to average over.
max_eval_steps = 100  # Max number of steps to take during each eval run.
q_lr = 0.0001  # Learning rate of the q net.
warmup_steps = 500  # For the first n number of steps, we will only sample randomly.
buffer_size = 2000  # Number of elements that can be stored in the buffer.
target_update = 200  # Number of iterations before updating Q target.
num_frames = 4  # Number of frames in frame stack.
max_skip_frames = 4  # Max number of frames to skip.
time_limit = 500  # Time limit before truncation.
bot_update = 500  # Number of iterations before updating the bot.
device = torch.device("cuda")


# Argument parsing
parser = ArgumentParser()
parser.add_argument("--eval", action="store_true")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()


class QNet(nn.Module):
    def __init__(
        self,
        obs_shape: torch.Size,
        action_count: int,
    ):
        nn.Module.__init__(self)
        channels = obs_shape[0] * obs_shape[1]  # Frames times channels
        self.net = nn.Sequential(
            nn.Conv2d(channels, 8, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 12, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(12, 64, 3, stride=2),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()
        self.advantage = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, action_count)
        )
        self.value = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.action_count = action_count
        init_orthogonal(self)

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, 1, 2)
        x = self.net(x)
        x = torch.max(torch.max(x, dim=3).values, dim=2).values
        x = self.net2(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)


env = SyncVectorEnv(
    [
        (
            lambda: TimeLimit(
                MFEnv(max_skip_frames=max_skip_frames, num_frames=num_frames),
                time_limit,
            )
        )
        for _ in range(num_envs)
    ]
)
test_env = MFEnv(max_skip_frames=max_skip_frames, num_frames=num_frames)

# If evaluating, load the latest policy
if args.eval:
    eval_done = False
    obs_space = env.single_observation_space
    act_space = env.single_action_space
    assert isinstance(obs_space, gym.spaces.Box)
    assert isinstance(act_space, gym.spaces.Discrete)
    q_net = QNet(torch.Size(obs_space.shape), int(act_space.n))
    q_net.load_state_dict(torch.load("temp/q_net.pt"))
    test_env = MFEnv(
        max_skip_frames=max_skip_frames,
        # render_mode="human",
        # view_channels=(0, 2, 4),
        num_frames=num_frames,
    )
    mcts_env = MFEnv(
        max_skip_frames=max_skip_frames,
        num_frames=num_frames,
        render_mode="human",
        view_channels=(0, 2, 4),
    )
    rollouts = 100
    with torch.no_grad():
        reward_total = 0.0
        obs_, info = test_env.reset()
        eval_obs = torch.from_numpy(np.array(obs_[0])).float()
        while True:
            state = test_env.state()

            bot_q_vals = q_net(
                torch.from_numpy(test_env.bot_obs()).float().unsqueeze(0)
            ).squeeze()
            bot_action = bot_q_vals.argmax(0).item()
            test_env.bot_step(bot_action)

            action = run_mcts(q_net, mcts_env, state, rollouts, discount, int(act_space.n))
            obs_, reward, eval_done, eval_trunc, _ = test_env.step(action)
            test_env.render()
            eval_obs = torch.from_numpy(np.array(obs_[0])).float()
            reward_total += reward
            if eval_done or eval_trunc:
                obs_, info = test_env.reset()
                eval_obs = torch.from_numpy(np.array(obs_)).float()
