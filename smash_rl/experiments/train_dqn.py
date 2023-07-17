"""
Trains an agent with a DQN.
"""
import copy
import random
from typing import Any, Mapping

import numpy as np  # type: ignore
import torch
import torch.nn as nn
import wandb
from gymnasium.wrappers.time_limit import TimeLimit
import gymnasium as gym
from tqdm import tqdm
from argparse import ArgumentParser

import smash_rl.conf
from smash_rl.algorithms.dqn import train_dqn_multi
from smash_rl.algorithms.replay_buffer import ReplayBuffer, StateReplayBuffer
from smash_rl.micro_fighter.env import MFEnv
from smash_rl.utils import init_orthogonal

_: Any
INF = 10**8

# Hyperparameters
train_steps = 128  # Number of steps to step through during sampling.
iterations = 10000  # Number of sample/train iterations.
train_iters = 8  # Number of passes over the samples collected.
train_batch_size = 128  # Minibatch size while training models.
discount = 0.99  # Discount factor applied to rewards.
q_epsilon = 0.9  # Epsilon for epsilon greedy strategy. This gets annealed over time.
eval_steps = 3  # Number of eval runs to average over.
max_eval_steps = 500  # Max number of steps to take during each eval run.
q_lr = 0.0003  # Learning rate of the q net.
warmup_steps = 500  # For the first n number of steps, we will only sample randomly.
buffer_size = 10000  # Number of elements that can be stored in the buffer.
target_update = 100  # Number of iterations before updating Q target.
num_frames = 4  # Number of frames in frame stack.
max_skip_frames = 1  # Max number of frames to skip.
time_limit = 500  # Time limit before truncation.
bot_update = 500  # Number of iterations before caching the current policy.
max_bots = 6  # Maximum number of bots to store.
start_elo = 1200  # Starting ELO score for each agent.
elo_k = 16  # ELO adjustment constant.
eval_every = 100  # Number of iterations before evaluating.
device = torch.device("cuda")


# Argument parsing
parser = ArgumentParser()
parser.add_argument("--eval", action="store_true")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()


class QNet(nn.Module):
    def __init__(
        self,
        obs_shape_spatial: torch.Size,
        obs_shape_stats: int,
        action_count: int,
    ):
        nn.Module.__init__(self)
        channels = obs_shape_spatial[0] * obs_shape_spatial[1]  # Frames times channels
        self.spatial_net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 256, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, stride=2),
            nn.ReLU(),
        )
        self.stats_net = nn.Sequential(
            nn.Linear(obs_shape_stats, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_count)
        )
        self.value = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.action_count = action_count
        init_orthogonal(self)

    def forward(self, spatial: torch.Tensor, stats: torch.Tensor):
        spatial = torch.flatten(spatial, 1, 2)
        spatial = self.spatial_net(spatial)
        spatial = torch.max(torch.max(spatial, dim=3).values, dim=2).values

        stats = self.stats_net(stats)

        x = torch.concat([spatial, stats], dim=1)
        x = self.net2(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)


env = TimeLimit(
    MFEnv(max_skip_frames=max_skip_frames, num_frames=num_frames),
    time_limit,
)
test_env = TimeLimit(MFEnv(max_skip_frames=max_skip_frames, num_frames=num_frames), max_eval_steps)

# If evaluating, load the latest policy
if args.eval:
    eval_done = False
    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(obs_space, gym.spaces.Tuple)
    assert isinstance(obs_space.spaces[0], gym.spaces.Box)
    assert isinstance(obs_space.spaces[1], gym.spaces.Box)
    spatial_obs_space = obs_space.spaces[0]
    stats_obs_space = obs_space.spaces[1]
    assert isinstance(act_space, gym.spaces.Discrete)
    q_net = QNet(
        torch.Size(spatial_obs_space.shape), stats_obs_space.shape[0], int(act_space.n)
    )
    q_net.load_state_dict(torch.load("temp/q_net.pt"))
    test_env = TimeLimit(
        MFEnv(
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
            bot_obs_1, bot_obs_2 = test_env.bot_obs()
            bot_q_vals = q_net(
                torch.from_numpy(bot_obs_1).float().unsqueeze(0),
                torch.from_numpy(bot_obs_2).float().unsqueeze(0),
            ).squeeze()
            bot_action = bot_q_vals.argmax(0).item()
            test_env.bot_step(bot_action)

            q_vals = q_net(eval_obs_1.unsqueeze(0), eval_obs_2.unsqueeze(0)).squeeze()
            action = q_vals.argmax(0).item()
            (obs_1_, obs_2_), reward, eval_done, eval_trunc, _ = test_env.step(action)
            test_env.render()
            eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
            eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
            if eval_done or eval_trunc:
                (obs_1_, obs_2_), _ = test_env.reset()
                eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()

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
assert isinstance(obs_space, gym.spaces.Tuple)
assert isinstance(obs_space.spaces[0], gym.spaces.Box)
assert isinstance(obs_space.spaces[1], gym.spaces.Box)
spatial_obs_space = obs_space.spaces[0]
stats_obs_space = obs_space.spaces[1]
assert isinstance(act_space, gym.spaces.Discrete)
assert isinstance(env.env, MFEnv)
q_net = QNet(
    torch.Size(spatial_obs_space.shape), stats_obs_space.shape[0], int(act_space.n)
)
if args.resume:
    q_net.load_state_dict(torch.load("temp/q_net.pt"))
q_net_target = copy.deepcopy(q_net)
q_net_target.to(device)
q_opt = torch.optim.Adam(q_net.parameters(), lr=q_lr)

# Bot Q networks
bot_data = [{"state_dict": q_net.state_dict(), "elo": start_elo}]
bot_q_net = copy.deepcopy(q_net)
bot_q_index = 0
bot_is_random = True
current_elo = start_elo

# A replay buffer stores experience collected over all sampling runs
buffer = ReplayBuffer(
    torch.Size(spatial_obs_space.shape),
    torch.Size((int(act_space.n),)),
    buffer_size,
)
stats_buffer = StateReplayBuffer(
    torch.Size(stats_obs_space.shape),
    buffer_size,
)

(obs_1_, obs_2_), _ = env.reset()
obs_1 = torch.from_numpy(obs_1_).float().unsqueeze(0)
obs_2 = torch.from_numpy(obs_2_).float().unsqueeze(0)
for step in tqdm(range(iterations), position=0):
    percent_done = step / iterations
    q_epsilon_real = q_epsilon * max(1.0 - percent_done, 0.05)
    env.env.set_dmg_reward_amount(1.0 - percent_done)

    # Collect experience
    with torch.no_grad():
        for _ in range(train_steps):
            # Choose bot action
            if bot_is_random:
                bot_action = random.randrange(0, int(act_space.n))
            else:
                bot_obs_1, bot_obs_2 = env.bot_obs()
                bot_q_vals = bot_q_net(
                    torch.from_numpy(bot_obs_1).float().unsqueeze(0),
                    torch.from_numpy(bot_obs_2).float().unsqueeze(0),
                ).squeeze()
                bot_action = bot_q_vals.argmax(0).item()
            env.bot_step(bot_action)

            # Choose player action
            if (
                random.random() < q_epsilon_real
                or step < warmup_steps
                or not buffer.filled
            ):
                action = random.randrange(0, int(act_space.n))
            else:
                q_vals = q_net(obs_1, obs_2)
                action = q_vals.argmax(1).item()

            try:
                (obs_1_, obs_2_), rewards, dones, truncs, _ = env.step(action)
                next_obs_1 = torch.from_numpy(obs_1_).float().unsqueeze(0)
                next_obs_2 = torch.from_numpy(obs_2_).float().unsqueeze(0)
                buffer.insert_step(
                    obs_1,
                    next_obs_1,
                    torch.tensor([action]).squeeze(0),
                    [rewards],
                    [dones],
                    None,
                    None,
                )
                stats_buffer.insert_step(obs_2, next_obs_2)
                obs_1 = next_obs_1
                obs_2 = next_obs_2

                if dones or truncs:
                    (obs_1_, obs_2_), _ = env.reset()
                    obs_1 = torch.from_numpy(obs_1_).float().unsqueeze(0)
                    obs_2 = torch.from_numpy(obs_2_).float().unsqueeze(0)
                    # Change opponent
                    if random.random() < 0.1:
                        bot_is_random = True
                    else:
                        bot_is_random = False
                        state_dict = random.choice(bot_data)["state_dict"]
                        assert isinstance(state_dict, Mapping)
                        bot_q_net.load_state_dict(state_dict)
            except KeyboardInterrupt:
                quit()
            except:
                obs = torch.Tensor(env.reset()[0]).float().unsqueeze(0)

    # Train
    if buffer.filled:
        total_q_loss = train_dqn_multi(
            q_net,
            q_net_target,
            q_opt,
            buffer,
            stats_buffer,
            device,
            train_iters,
            train_batch_size,
            discount,
        )

        # Evaluate the network's performance.
        if (step + 1) % eval_every == 0:
            eval_done = False
            with torch.no_grad():
                (obs_1_, obs_2_), _ = test_env.reset()
                eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
                eval_bot_q_net = QNet(
                    torch.Size(spatial_obs_space.shape),
                    stats_obs_space.shape[0],
                    int(act_space.n),
                )
                for _ in range(eval_steps):
                    i = random.randrange(0, len(bot_data))
                    state_dict = bot_data[i]["state_dict"]
                    assert isinstance(state_dict, Mapping)
                    eval_bot_q_net.load_state_dict(state_dict)
                    b_elo = bot_data[i]["elo"]
                    assert isinstance(b_elo, int)
                    try:
                        for _ in range(max_eval_steps):
                            bot_obs_1, bot_obs_2 = test_env.bot_obs()
                            bot_q_vals = bot_q_net(
                                torch.from_numpy(bot_obs_1).float().unsqueeze(0),
                                torch.from_numpy(bot_obs_2).float().unsqueeze(0),
                            ).squeeze()
                            bot_action = bot_q_vals.argmax(0).item()
                            test_env.bot_step(bot_action)

                            q_vals = q_net(
                                eval_obs_1.unsqueeze(0), eval_obs_2.unsqueeze(0)
                            ).squeeze()
                            action = q_vals.argmax(0).item()
                            (
                                (obs_1_, obs_2_),
                                reward,
                                eval_done,
                                eval_trunc,
                                _,
                            ) = test_env.step(action)
                            eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                            eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
                            if eval_done or eval_trunc:
                                if reward > 0.5:
                                    # Current network won
                                    a = 1.0
                                    b = 0.0
                                elif reward < -0.5:
                                    # Opponent won
                                    b = 1.0
                                    a = 0.0
                                else:
                                    # They tied
                                    a = 0.5
                                    b = 0.5
                                ea = 1.0 / (
                                    1.0 + 10.0 ** ((b_elo - current_elo) / 400.0)
                                )
                                eb = 1.0 / (
                                    1.0 + 10.0 ** ((current_elo - b_elo) / 400.0)
                                )
                                current_elo = current_elo + elo_k * (a - ea)
                                bot_data[i]["elo"] = int(b_elo + elo_k * (a - ea))
                                (obs_1_, obs_2_), info = test_env.reset()
                                eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                                eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
                                break
                    except KeyboardInterrupt:
                        quit()
                    except:
                        pass
            wandb.log(
                {
                    "current_elo": current_elo,
                },
            )

        wandb.log(
            {
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
            if len(bot_data) < max_bots:
                bot_data.append(
                    {"state_dict": q_net.state_dict(), "elo": int(current_elo)}
                )
            else:
                bot_q_index = (bot_q_index + 1) % max_bots
                bot_data[bot_q_index] = {
                    "state_dict": q_net.state_dict(),
                    "elo": int(current_elo),
                }

        # Save
        if (step + 1) % 10 == 0:
            torch.save(q_net.state_dict(), "temp/q_net.pt")
