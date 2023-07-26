"""
Trains an agent with PPO, augmented with retrieval.
"""
from argparse import ArgumentParser
import copy
import math
import random
import time
from typing import Any, Mapping
import gymnasium as gym
import numpy as np
from sklearn.decomposition import PCA  # type: ignore
import pickle
from horapy import HNSWIndex  # type: ignore

import torch
import torch.nn as nn
import wandb
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.normalize import NormalizeReward
from torch.distributions import Categorical
from tqdm import tqdm

from smash_rl.algorithms.ppo_multi import train_ppo
from smash_rl.algorithms.rollout_buffer import RolloutBuffer, StateRolloutBuffer
from smash_rl.conf import entity
from smash_rl.micro_fighter.env import RetrievalMFEnv, RetrievalContext
from smash_rl.utils import init_orthogonal
from smash_rl_rust import test_jit
import json

from smash_rl_rust import RolloutContext

_: Any

# Hyperparameters
num_envs = 32  # Number of environments to step through at once during sampling.
train_steps = 64  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs.
iterations = 1000  # Number of sample/train iterations.
train_iters = 2  # Number of passes over the samples collected.
train_batch_size = 128  # Minibatch size while training models.
discount = 0.995  # Discount factor applied to rewards.
lambda_ = 0.95  # Lambda for GAE.
epsilon = 0.2  # Epsilon for importance sample clipping.
v_lr = 0.001  # Learning rate of the value net.
p_lr = 0.0003  # Learning rate of the policy net.
num_frames = 4  # Number of frames in frame stack.
max_skip_frames = 1  # Max number of frames to skip.
time_limit = 1000  # Time limit before truncation.
bot_update = 20  # Number of iterations before caching the current policy.
max_bots = 10  # Maximum number of bots to store.
start_elo = 1200  # Starting ELO score for each agent.
elo_k = 16  # ELO adjustment constant.
eval_every = 2  # Number of iterations before evaluating.
eval_steps = 5  # Number of eval runs to perform.
max_eval_steps = 500  # Max number of steps to take during each eval run.
num_workers = 8
top_k = 4
device = torch.device("cuda")  # Device to use during training.

# Argument parsing
parser = ArgumentParser()
parser.add_argument("--eval", action="store_true")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()


class NeighborEncoder(nn.Module):
    """
    Encodes neighbor information.
    """

    def __init__(
        self,
        obs_embedding_size: int,
        neighbor_shape_spatial: torch.Size,
        neighbor_shape_scalar: int,
    ):
        nn.Module.__init__(self)
        channels = neighbor_shape_spatial[1]
        self.spatial_net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2),
            nn.ReLU(),
        )
        self.scalar_net = nn.Sequential(
            nn.Linear(neighbor_shape_scalar + obs_embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

    def forward(
        self, obs_embedding: torch.Tensor, spatial: torch.Tensor, scalar: torch.Tensor
    ):
        batch_size = spatial.shape[0]
        k = spatial.shape[1]

        spatial = torch.flatten(spatial, 0, 1)
        spatial = self.spatial_net(spatial)
        spatial = torch.max(torch.max(spatial, dim=3).values, dim=2).values

        scalar = torch.flatten(
            torch.concatenate([scalar, obs_embedding.unsqueeze(1).repeat([1, k, 1])], 2),
            0,
            1,
        )
        scalar = self.scalar_net(scalar)

        x = torch.concat([spatial, scalar], dim=1)
        x = self.net2(x)

        x = torch.reshape(x, (batch_size, k, 512))
        x = x.sum(1, keepdim=False) / math.sqrt(k)

        return x


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
    def __init__(
        self,
        obs_shape_spatial: torch.Size,
        obs_shape_stats: int,
        neighbor_shape_spatial: torch.Size,
        neighbor_shape_scalar: int,
    ):
        nn.Module.__init__(self)
        self.shared = SharedNet(obs_shape_spatial, obs_shape_stats)
        self.neighbor = NeighborEncoder(512, neighbor_shape_spatial, neighbor_shape_scalar)
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        init_orthogonal(self)

    def forward(
        self,
        spatial: torch.Tensor,
        stats: torch.Tensor,
        neighbor_spatial: torch.Tensor,
        neighbor_scalar: torch.Tensor,
    ):
        x = self.shared(spatial, stats)
        neighbor_x = self.neighbor(x, neighbor_spatial, neighbor_scalar)
        x = torch.concat([x, neighbor_x], 1)
        x = self.net(x)
        return x


class PolicyNet(nn.Module):
    def __init__(
        self,
        obs_shape_spatial: torch.Size,
        obs_shape_stats: int,
        action_count: int,
        neighbor_shape_spatial: torch.Size,
        neighbor_shape_scalar: int,
    ):
        nn.Module.__init__(self)
        self.shared = SharedNet(obs_shape_spatial, obs_shape_stats)
        self.neighbor = NeighborEncoder(512, neighbor_shape_spatial, neighbor_shape_scalar)
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_count),
            nn.LogSoftmax(1),
        )
        init_orthogonal(self)

    def forward(
        self,
        spatial: torch.Tensor,
        stats: torch.Tensor,
        neighbor_spatial: torch.Tensor,
        neighbor_scalar: torch.Tensor,
    ):
        x = self.shared(spatial, stats)
        neighbor_x = self.neighbor(x, neighbor_spatial, neighbor_scalar)
        x = torch.concat([x, neighbor_x], 1)
        x = self.net(x)
        return x


encoder = torch.jit.load("temp/encoder.ptc")
with open("temp/pca.pkl", "rb") as rfile:
    pca = pickle.load(rfile)
retrieval_ctx = RetrievalContext(
    128, "temp/index.bin", "temp/generated", "temp/episode_data.json", encoder, pca
)
env = SyncVectorEnv(
    [
        lambda: TimeLimit(
            NormalizeReward(
                RetrievalMFEnv(
                    retrieval_ctx,
                    top_k,
                    max_skip_frames=max_skip_frames,
                    num_frames=num_frames,
                )
            ),
            time_limit,
        )
        for _ in range(num_envs)
    ]
)
test_env = TimeLimit(
    RetrievalMFEnv(
        retrieval_ctx, top_k, max_skip_frames=max_skip_frames, num_frames=num_frames
    ),
    max_eval_steps,
)
if __name__ == "__main__":
    # If evaluating, load the latest policy
    if args.eval:
        eval_done = False
        obs_space = test_env.observation_space
        act_space = test_env.action_space
        assert isinstance(obs_space, gym.spaces.Tuple)
        assert isinstance(obs_space.spaces[0], gym.spaces.Box)
        assert isinstance(obs_space.spaces[1], gym.spaces.Box)
        assert isinstance(obs_space.spaces[2], gym.spaces.Box)
        assert isinstance(obs_space.spaces[3], gym.spaces.Box)
        spatial_obs_space = obs_space.spaces[0]
        stats_obs_space = obs_space.spaces[1]
        neighbor_spatial_obs_space = obs_space.spaces[2]
        neighbor_scalar_obs_space = obs_space.spaces[3]
        assert isinstance(act_space, gym.spaces.Discrete)
        p_net = PolicyNet(
            torch.Size(spatial_obs_space.shape),
            stats_obs_space.shape[0],
            int(act_space.n),
            torch.Size(neighbor_spatial_obs_space.shape),
            neighbor_scalar_obs_space.shape[1],
        )
        # p_net.load_state_dict(torch.load("temp/p_net_retrieval.pt"))
        test_env = TimeLimit(
            RetrievalMFEnv(
                retrieval_ctx,
                top_k,
                max_skip_frames=max_skip_frames,
                render_mode="human",
                view_channels=(0, 2, 3),
                num_frames=num_frames,
            ),
            max_eval_steps,
        )
        with torch.no_grad():
            (obs_1_, obs_2_, n_obs_1_, n_obs_2_), _ = test_env.reset()
            eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
            eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
            eval_n_obs_1 = torch.from_numpy(np.array(n_obs_1_)).float()
            eval_n_obs_2 = torch.from_numpy(np.array(n_obs_2_)).float()
            while True:
                bot_obs_1, bot_obs_2, bot_n_obs_1, bot_n_obs_2 = test_env.bot_obs()
                bot_action_probs = p_net(
                    torch.from_numpy(bot_obs_1).unsqueeze(0).float(),
                    torch.from_numpy(bot_obs_2).unsqueeze(0).float(),
                    torch.from_numpy(bot_n_obs_1).unsqueeze(0).float(),
                    torch.from_numpy(bot_n_obs_2).unsqueeze(0).float(),
                ).squeeze()
                bot_action = Categorical(logits=bot_action_probs).sample().numpy()
                test_env.bot_step(bot_action)

                action_probs = p_net(
                    eval_obs_1.unsqueeze(0),
                    eval_obs_2.unsqueeze(0),
                    eval_n_obs_1.unsqueeze(0),
                    eval_n_obs_2.unsqueeze(0),
                ).squeeze()
                action = Categorical(logits=action_probs).sample().numpy()
                (
                    (obs_1_, obs_2_, n_obs_1_, n_obs_2_),
                    reward,
                    eval_done,
                    eval_trunc,
                    eval_info,
                ) = test_env.step(action)

                test_env.render()
                eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
                eval_n_obs_1 = torch.from_numpy(np.array(n_obs_1_)).float()
                eval_n_obs_2 = torch.from_numpy(np.array(n_obs_2_)).float()
                if eval_done or eval_trunc:
                    (obs_1_, obs_2_, n_obs_1_, n_obs_2_), _ = test_env.reset()
                    eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                    eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
                    eval_n_obs_1 = torch.from_numpy(np.array(n_obs_1_)).float()
                    eval_n_obs_2 = torch.from_numpy(np.array(n_obs_2_)).float()

    wandb.init(
        project="smash-rl",
        entity=entity,
        config={
            "experiment": "micro fighter ppo with retrieval",
            "num_envs": num_envs,
            "train_steps": train_steps,
            "train_iters": train_iters,
            "train_batch_size": train_batch_size,
            "discount": discount,
            "lambda": lambda_,
            "epsilon": epsilon,
            "v_lr": v_lr,
            "p_lr": p_lr,
            "top_k": top_k,
        },
    )

    # Initialize policy and value networks
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    assert isinstance(obs_space, gym.spaces.Tuple)
    assert isinstance(obs_space.spaces[0], gym.spaces.Box)
    assert isinstance(obs_space.spaces[1], gym.spaces.Box)
    assert isinstance(obs_space.spaces[2], gym.spaces.Box)
    assert isinstance(obs_space.spaces[3], gym.spaces.Box)
    spatial_obs_space = obs_space.spaces[0]
    stats_obs_space = obs_space.spaces[1]
    neighbor_spatial_obs_space = obs_space.spaces[2]
    neighbor_scalar_obs_space = obs_space.spaces[3]
    assert isinstance(act_space, gym.spaces.Discrete)
    v_net = ValueNet(
        torch.Size(spatial_obs_space.shape),
        stats_obs_space.shape[0],
        torch.Size(neighbor_spatial_obs_space.shape),
        neighbor_scalar_obs_space.shape[1],
    )
    p_net = PolicyNet(
        torch.Size(spatial_obs_space.shape),
        stats_obs_space.shape[0],
        int(act_space.n),
        torch.Size(neighbor_spatial_obs_space.shape),
        neighbor_scalar_obs_space.shape[1],
    )
    if args.resume:
        p_net.load_state_dict(torch.load("temp/p_net_retrieval.pt"))
        v_net.load_state_dict(torch.load("temp/v_net_retrieval.pt"))
    v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
    p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)

    # Bot Q networks
    bot_data = [{"state_dict": p_net.state_dict(), "elo": start_elo}]
    bot_nets = [copy.deepcopy(p_net)]
    bot_p_indices = [0 for _ in range(num_envs)]
    current_elo = start_elo

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
    buffer_n_spatial = StateRolloutBuffer(
        torch.Size(neighbor_spatial_obs_space.shape), num_envs, train_steps
    )
    buffer_n_scalar = StateRolloutBuffer(
        torch.Size(neighbor_scalar_obs_space.shape), num_envs, train_steps
    )

    (obs_1_, obs_2_, n_obs_1_, n_obs_2_), _ = env.reset()
    obs_1 = torch.from_numpy(obs_1_).float()
    obs_2 = torch.from_numpy(obs_2_).float()
    n_obs_1 = torch.from_numpy(n_obs_1_).float()
    n_obs_2 = torch.from_numpy(n_obs_2_).float()
    for step in tqdm(range(iterations), position=0):
        percent_done = step / iterations
        for env_ in env.envs:
            assert isinstance(env_, TimeLimit)
            assert isinstance(env_.env, NormalizeReward)
            assert isinstance(env_.env.env, RetrievalMFEnv)
            env_.env.set_dmg_reward_amount(1.0 - percent_done)

        # Collect experience
        with torch.no_grad():
            total_entropy = 0.0
            for _ in tqdm(range(train_steps), position=1):
                # Choose bot action
                for env_index, env_ in enumerate(env.envs):
                    assert isinstance(env_, TimeLimit)
                    assert isinstance(env_.env, NormalizeReward)
                    assert isinstance(env_.env.env, RetrievalMFEnv)
                    (
                        bot_obs_1,
                        bot_obs_2,
                        bot_n_obs_1,
                        bot_n_obs_2,
                    ) = env_.env.env.bot_obs()
                    bot_action_probs = bot_nets[bot_p_indices[env_index]](
                        torch.from_numpy(bot_obs_1).float().unsqueeze(0),
                        torch.from_numpy(bot_obs_2).float().unsqueeze(0),
                        torch.from_numpy(bot_n_obs_1).float().unsqueeze(0),
                        torch.from_numpy(bot_n_obs_2).float().unsqueeze(0),
                    ).squeeze(0)
                    bot_action = Categorical(logits=bot_action_probs).sample().item()
                    env_.env.bot_step(bot_action)

                # Choose player action
                action_probs = p_net(obs_1, obs_2, n_obs_1, n_obs_2)
                action_distr = Categorical(logits=action_probs)
                total_entropy += action_distr.entropy().mean()
                actions = action_distr.sample().numpy()

                try:
                    (
                        (obs_1_, obs_2_, n_obs_1_, n_obs_2_),
                        rewards,
                        dones,
                        truncs,
                        _,
                    ) = env.step(actions)
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
                    buffer_n_spatial.insert_step(n_obs_1)
                    buffer_n_scalar.insert_step(n_obs_2)
                    obs_1 = torch.from_numpy(obs_1_).float()
                    obs_2 = torch.from_numpy(obs_2_).float()
                    n_obs_1 = torch.from_numpy(n_obs_1_).float()
                    n_obs_2 = torch.from_numpy(n_obs_2_).float()

                    # Change opponent when environment ends
                    for env_index, env_ in enumerate(env.envs):
                        if dones[env_index] or truncs[env_index]:
                            state_dict = random.choice(bot_data)["state_dict"]
                            assert isinstance(state_dict, Mapping)
                            bot_p_indices[env_index] = random.randrange(
                                0, len(bot_data)
                            )
                            bot_nets[bot_p_indices[env_index]].load_state_dict(
                                state_dict
                            )

                except KeyboardInterrupt:
                    quit()
                except Exception as e:
                    print(f"Exception in simulation: {e}. Resetting.")
                    (obs_1_, obs_2_, n_obs_1_, n_obs_2_), _ = env.reset()
                    obs_1 = torch.from_numpy(obs_1_).float()
                    obs_2 = torch.from_numpy(obs_2_).float()
                    n_obs_1 = torch.from_numpy(n_obs_1_).float()
                    n_obs_2 = torch.from_numpy(n_obs_2_).float()
        buffer_spatial.insert_final_step(obs_1)
        buffer_stats.insert_final_step(obs_2)

        # Train
        total_p_loss, total_v_loss = train_ppo(
            p_net,
            v_net,
            p_opt,
            v_opt,
            buffer_spatial,
            [buffer_stats, buffer_n_spatial, buffer_n_scalar],
            device,
            train_iters,
            train_batch_size,
            discount,
            lambda_,
            epsilon,
            entropy_coeff=0.0001,
        )
        buffer_spatial.clear()
        buffer_stats.clear()
        buffer_n_spatial.clear()
        buffer_n_scalar.clear()

        # Evaluate the network's performance.
        if (step + 1) % eval_every == 0:
            eval_done = False
            with torch.no_grad():
                (obs_1_, obs_2_, n_obs_1_, n_obs_2_), _ = test_env.reset()
                eval_obs_1 = torch.from_numpy(obs_1_).float()
                eval_obs_2 = torch.from_numpy(obs_2_).float()
                eval_n_obs_1 = torch.from_numpy(n_obs_1_).float()
                eval_n_obs_2 = torch.from_numpy(n_obs_2_).float()
                for _ in range(eval_steps):
                    eval_bot_index = random.randrange(0, len(bot_data))
                    b_elo = bot_data[eval_bot_index]["elo"]
                    assert isinstance(b_elo, int)
                    try:
                        for _ in range(max_eval_steps):
                            (
                                bot_obs_1,
                                bot_obs_2,
                                bot_n_obs_1,
                                bot_n_obs_2,
                            ) = test_env.bot_obs()
                            bot_action_probs = bot_nets[eval_bot_index](
                                torch.from_numpy(bot_obs_1).unsqueeze(0).float(),
                                torch.from_numpy(bot_obs_2).unsqueeze(0).float(),
                                torch.from_numpy(bot_n_obs_1).unsqueeze(0).float(),
                                torch.from_numpy(bot_n_obs_2).unsqueeze(0).float(),
                            ).squeeze()
                            bot_action = (
                                Categorical(logits=bot_action_probs).sample().numpy()
                            )
                            test_env.bot_step(bot_action)

                            action_probs = p_net(
                                eval_obs_1.unsqueeze(0),
                                eval_obs_2.unsqueeze(0),
                                eval_n_obs_1.unsqueeze(0),
                                eval_n_obs_2.unsqueeze(0),
                            ).squeeze()
                            actions = Categorical(logits=action_probs).sample().numpy()
                            (
                                (obs_1_, obs_2_, n_obs_1_, n_obs_2_),
                                reward,
                                eval_done,
                                eval_trunc,
                                eval_info,
                            ) = test_env.step(actions)

                            eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                            eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
                            eval_n_obs_1 = torch.from_numpy(n_obs_1_).float()
                            eval_n_obs_2 = torch.from_numpy(n_obs_2_).float()
                            if eval_done or eval_trunc:
                                if eval_done:
                                    if eval_info["player_won"]:
                                        # Current network won
                                        a = 1.0
                                        b = 0.0
                                    else:
                                        # Opponent won
                                        b = 1.0
                                        a = 0.0
                                elif eval_trunc:
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
                                bot_data[eval_bot_index]["elo"] = int(
                                    b_elo + elo_k * (b - eb)
                                )
                                (
                                    obs_1_,
                                    obs_2_,
                                    n_obs_1_,
                                    n_obs_2_,
                                ), info = test_env.reset()
                                eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                                eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
                                eval_n_obs_1 = torch.from_numpy(n_obs_1_).float()
                                eval_n_obs_2 = torch.from_numpy(n_obs_2_).float()
                                break
                    except KeyboardInterrupt:
                        quit()
                    except Exception as e:
                        print(f"Exception during evaluation: {e}.")
                        pass
            wandb.log(
                {
                    "current_elo": current_elo,
                },
            )

        wandb.log(
            {
                "avg_v_loss": total_v_loss / train_iters,
                "avg_p_loss": total_p_loss / train_iters,
                "entropy": total_entropy / (num_envs * train_iters),
            }
        )

        # Update bot nets
        if (step + 1) % bot_update == 0:
            if len(bot_data) < max_bots:
                bot_data.append(
                    {"state_dict": p_net.state_dict(), "elo": int(current_elo)}
                )
                bot_nets.append(copy.deepcopy(p_net))
            else:
                # Replace bot with lowest ELO if that ELO is less than the current ELO
                next_bot_index = 0
                lowest_elo = current_elo
                for bot_index, data in enumerate(bot_data):
                    assert isinstance(data["elo"], int)
                    if data["elo"] < lowest_elo:
                        next_bot_index = bot_index
                if current_elo > lowest_elo:
                    bot_data[next_bot_index] = {
                        "state_dict": p_net.state_dict(),
                        "elo": int(current_elo),
                    }
                    bot_nets[next_bot_index].load_state_dict(p_net.state_dict())

        torch.save(p_net.state_dict(), "temp/p_net_retrieval.pt")
        torch.save(v_net.state_dict(), "temp/v_net_retrieval.pt")
