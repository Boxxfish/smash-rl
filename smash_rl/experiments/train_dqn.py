"""
Trains an agent with a DQN.
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

_: Any
INF = 10**8

# Hyperparameters
train_steps = 32  # Number of steps to step through during sampling.
iterations = 10000  # Number of sample/train iterations.
train_iters = 4  # Number of passes over the samples collected.
train_batch_size = 64  # Minibatch size while training models.
discount = 0.99  # Discount factor applied to rewards.
q_epsilon = 0.9  # Epsilon for epsilon greedy strategy. This gets annealed over time.
eval_steps = 1  # Number of eval runs to average over.
max_eval_steps = 100  # Max number of steps to take during each eval run.
q_lr = 0.0001  # Learning rate of the q net.
warmup_steps = 500  # For the first n number of steps, we will only sample randomly.
buffer_size = 10000  # Number of elements that can be stored in the buffer.
target_update = 100  # Number of iterations before updating Q target.
num_frames = 4  # Number of frames in frame stack.
max_skip_frames = 4  # Max number of frames to skip.
time_limit = 500  # Time limit before truncation.
bot_update = 500  # Number of iterations before caching the current policy.
max_bots = 6 # Maximum number of iterations.
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
            nn.Conv2d(channels, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_count)
        )
        self.value = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
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


env = TimeLimit(
    MFEnv(max_skip_frames=max_skip_frames, num_frames=num_frames),
    time_limit,
)
test_env = MFEnv(max_skip_frames=max_skip_frames, num_frames=num_frames)

# If evaluating, load the latest policy
if args.eval:
    eval_done = False
    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(obs_space, gym.spaces.Box)
    assert isinstance(act_space, gym.spaces.Discrete)
    q_net = QNet(torch.Size(obs_space.shape), int(act_space.n))
    q_net.load_state_dict(torch.load("temp/q_net.pt"))
    test_env = MFEnv(
        max_skip_frames=max_skip_frames,
        render_mode="human",
        view_channels=(0, 2, 4),
        num_frames=num_frames,
    )
    with torch.no_grad():
        reward_total = 0.0
        obs_, info = test_env.reset()
        eval_obs = torch.from_numpy(np.array(obs_)).float()
        while True:
            bot_q_vals = q_net(
                torch.from_numpy(test_env.bot_obs()).float().unsqueeze(0)
            ).squeeze()
            bot_action = bot_q_vals.argmax(0).item()
            test_env.bot_step(bot_action)

            q_vals = q_net(eval_obs.unsqueeze(0)).squeeze()
            action = q_vals.argmax(0).item()
            obs_, reward, eval_done, eval_trunc, _ = test_env.step(action)
            test_env.render()
            eval_obs = torch.from_numpy(np.array(obs_)).float()
            reward_total += reward
            if eval_done or eval_trunc:
                obs_, info = test_env.reset()
                eval_obs = torch.from_numpy(np.array(obs_)).float()

wandb.init(
    project="smash-rl",
    entity=smash_rl.conf.entity,
    config={
        "experiment": "micro fighter dqn",
        "train_steps": train_steps,
        "train_iters": train_iters,
        "train_batch_size": train_batch_size,
        "discount": discount,
        "initial_epsilon": q_epsilon,
        "max_eval_steps": max_eval_steps,
        "q_lr": q_lr,
        "num_frames": num_frames,
        "max_skip_frames": max_skip_frames,
        "bot_update": bot_update,
        "target_update": target_update,
    },
)

# Initialize Q network
obs_space = env.observation_space
act_space = env.action_space
assert isinstance(obs_space, gym.spaces.Box)
assert isinstance(act_space, gym.spaces.Discrete)
assert isinstance(env.env, MFEnv)
q_net = QNet(torch.Size(obs_space.shape), int(act_space.n))
q_net_target = copy.deepcopy(q_net)
q_net_target.to(device)
q_opt = torch.optim.Adam(q_net.parameters(), lr=q_lr)

# Bot Q networks
bot_q_nets = [q_net.state_dict()]
bot_q_net = copy.deepcopy(q_net)
bot_q_index = 0

# A replay buffer stores experience collected over all sampling runs
buffer = ReplayBuffer(
    torch.Size(obs_space.shape),
    torch.Size((int(act_space.n),)),
    buffer_size,
)

obs = torch.Tensor(env.reset()[0]).float().unsqueeze(0)
for step in tqdm(range(iterations), position=0):
    percent_done = step / iterations
    q_epsilon_real = q_epsilon * max(1.0 - percent_done, 0.05)
    env.env.set_dmg_reward_amount(1.0 - percent_done)

    # Collect experience
    with torch.no_grad():
        for _ in range(train_steps):
            bot_q_vals = bot_q_net(
                torch.from_numpy(env.bot_obs()).float().unsqueeze(0)
            ).squeeze()
            bot_action = bot_q_vals.argmax(0).item()
            env.bot_step(bot_action)

            if random.random() < q_epsilon_real or step < warmup_steps:
                action = random.randrange(0, int(act_space.n))
            else:
                q_vals = q_net(obs)
                action = q_vals.argmax(1).item()
            try:
                obs_, rewards, dones, truncs, _ = env.step(action)
                next_obs = torch.from_numpy(obs_).float().unsqueeze(0)
                buffer.insert_step(
                    obs,
                    next_obs,
                    torch.tensor([action]).squeeze(0),
                    [rewards],
                    [dones],
                    None,
                    None,
                )
                obs = next_obs

                if dones or truncs:
                    obs = torch.Tensor(env.reset()[0]).float().unsqueeze(0)
                    # Change opponent
                    bot_q_net.load_state_dict(random.choice(bot_q_nets))
            except:
                obs = torch.Tensor(env.reset()[0]).float().unsqueeze(0)

    # Train
    if buffer.filled:
        total_q_loss = train_dqn(
            q_net,
            q_net_target,
            q_opt,
            buffer,
            device,
            train_iters,
            train_batch_size,
            discount,
        )

        # Evaluate the network's performance after this training iteration.
        # eval_done = False
        # with torch.no_grad():
        #     reward_total = 0.0
        #     pred_reward_total = 0
        #     obs_, info = test_env.reset()
        #     eval_obs = torch.from_numpy(np.array(obs_)).float()
        #     for _ in range(eval_steps):
        #         try:
        #             steps_taken = 0
        #             score = 0
        #             for _ in range(max_eval_steps):
        #                 bot_q_vals = bot_q_net(
        #                     torch.from_numpy(test_env.bot_obs()).float().unsqueeze(0)
        #                 ).squeeze()
        #                 bot_action = bot_q_vals.argmax(0).item()
        #                 test_env.bot_step(bot_action)

        #                 q_vals = q_net(eval_obs.unsqueeze(0)).squeeze()
        #                 action = q_vals.argmax(0).item()
        #                 pred_reward_total += (
        #                     q_net(eval_obs.unsqueeze(0)).squeeze().max(0).values.item()
        #                 )
        #                 obs_, reward, eval_done, eval_trunc, _ = test_env.step(action)
        #                 eval_obs = torch.from_numpy(np.array(obs_)).float()
        #                 steps_taken += 1
        #                 reward_total += reward
        #                 if eval_done or eval_trunc:
        #                     obs_, info = test_env.reset()
        #                     eval_obs = torch.from_numpy(np.array(obs_)).float()
        #                     break
        #         except:
        #             pass

        wandb.log(
            {
                "avg_eval_episode_reward": reward_total / eval_steps,
                # "avg_eval_episode_predicted_reward": pred_reward_total / eval_steps,
                "avg_q_loss": total_q_loss / train_iters,
                "q_lr": q_opt.param_groups[-1]["lr"],
                "epsilon": q_epsilon_real,
            }
        )

        # Update Q target
        if (step + 1) % target_update == 0:
            q_net_target.load_state_dict(q_net.state_dict())

        # Update bot Q nets
        if (step + 1) % bot_update == 0:
            if len(bot_q_nets) < max_bots:
                bot_q_nets.append(q_net.state_dict())
            else:
                bot_q_index = (bot_q_index + 1) % max_bots
                bot_q_nets[bot_q_index] = q_net.state_dict()

        # Save
        if (step + 1) % 10 == 0:
            torch.save(q_net.state_dict(), "temp/q_net.pt")
