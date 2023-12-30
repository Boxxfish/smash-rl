"""
Trains an agent with PPO, augmented with retrieval.
"""
from argparse import ArgumentParser
import copy
import math
import time
from typing import Any
import gymnasium as gym
import numpy as np
from sklearn.decomposition import PCA  # type: ignore
import pickle

import torch
import torch.nn as nn
import wandb
from gymnasium.wrappers.time_limit import TimeLimit
from torch.distributions import Categorical
from tqdm import tqdm

from smash_rl.algorithms.ppo_multi import train_ppo
from smash_rl.algorithms.rollout_buffer import RolloutBuffer, StateRolloutBuffer
from smash_rl.conf import entity
from smash_rl.micro_fighter.env import RetrievalMFEnv, RetrievalContext
from smash_rl.utils import init_orthogonal

from smash_rl_rust import RolloutContext

_: Any

# Hyperparameters
num_envs = 64  # Number of environments to step through at once during sampling.
train_steps = 64  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs.
iterations = 1000  # Number of sample/train iterations.
train_iters = 2  # Number of passes over the samples collected.
train_batch_size = 512  # Minibatch size while training models.
discount = 0.995  # Discount factor applied to rewards.
lambda_ = 0.95  # Lambda for GAE.
epsilon = 0.2  # Epsilon for importance sample clipping.
v_lr = 0.001  # Learning rate of the value net.
p_lr = 0.0003  # Learning rate of the policy net.
num_frames = 4  # Number of frames in frame stack.
max_skip_frames = 2  # Max number of frames to skip.
time_limit = 1000  # Time limit before truncation.
bot_update = 20  # Number of iterations before caching the current policy.
max_bots = 10  # Maximum number of bots to store.
start_elo = 1200  # Starting ELO score for each agent.
elo_k = 16  # ELO adjustment constant.
eval_every = 4  # Number of iterations before evaluating.
eval_steps = 5  # Number of eval runs to perform.
max_eval_steps = 500  # Max number of steps to take during each eval run.
num_workers = 8
top_k = 4
entropy_coeff = 0.003
device = torch.device("cuda")  # Device to use during training.
use_neighbors = True # Whether networks actually incorporate neighbor observations.

# Argument parsing
parser = ArgumentParser()
parser.add_argument("--eval", action="store_true")
parser.add_argument("--compare-retrieval", action="store_true")
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
            nn.Conv2d(channels, 32, 5),
            nn.ReLU(),
            nn.Conv2d(32, 256, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(256, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
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
            nn.Linear(128 + 400, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

    def forward(
        self, obs_embedding: torch.Tensor, spatial: torch.Tensor, scalar: torch.Tensor
    ):
        batch_size = spatial.shape[0]
        k = spatial.shape[1]

        spatial_shape = list(spatial.shape)
        scalar_shape = list(scalar.shape)

        spatial = spatial.reshape(
            [spatial_shape[0] * spatial_shape[1]] + spatial_shape[2:]
        )
        scalar = scalar.reshape([scalar_shape[0] * scalar_shape[1]] + scalar_shape[2:])

        spatial = self.spatial_net(spatial)

        scalar = torch.concatenate([scalar, obs_embedding.repeat([k, 1])], 1)
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
            nn.Conv2d(channels, 32, 5),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 16, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
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
            nn.Linear(256 + 400, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

    def forward(self, spatial: torch.Tensor, stats: torch.Tensor):
        spatial = torch.flatten(spatial, 1, 2)
        spatial = self.spatial_net(spatial)

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
        use_neighbors: bool,
    ):
        nn.Module.__init__(self)
        self.shared = SharedNet(obs_shape_spatial, obs_shape_stats)
        self.neighbor = NeighborEncoder(
            512, neighbor_shape_spatial, neighbor_shape_scalar
        )
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.use_neighbors = use_neighbors
        init_orthogonal(self)

    def forward(
        self,
        spatial: torch.Tensor,
        stats: torch.Tensor,
        neighbor_spatial: torch.Tensor,
        neighbor_scalar: torch.Tensor,
    ):
        x = self.shared(spatial, stats)

        if self.use_neighbors:
            neighbor_x = self.neighbor(x, neighbor_spatial, neighbor_scalar)
        else:
            batch_size = x.shape[0]
            neighbor_x = torch.zeros([batch_size, 512])

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
        use_neighbors: bool,
    ):
        nn.Module.__init__(self)
        self.shared = SharedNet(obs_shape_spatial, obs_shape_stats)
        self.neighbor = NeighborEncoder(
            512, neighbor_shape_spatial, neighbor_shape_scalar
        )
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_count),
            nn.LogSoftmax(1),
        )
        self.use_neighbors = use_neighbors
        init_orthogonal(self)

    def forward(
        self,
        spatial: torch.Tensor,
        stats: torch.Tensor,
        neighbor_spatial: torch.Tensor,
        neighbor_scalar: torch.Tensor,
    ):
        x = self.shared(spatial, stats)

        if self.use_neighbors:
            neighbor_x = self.neighbor(x, neighbor_spatial, neighbor_scalar)
        else:
            batch_size = x.shape[0]
            neighbor_x = torch.zeros([batch_size, 512])

        x = torch.concat([x, neighbor_x], 1)
        x = self.net(x)
        return x


encoder = torch.jit.load("temp/encoder.ptc")
with open("temp/pca.pkl", "rb") as rfile:
    pca = pickle.load(rfile)
retrieval_ctx = RetrievalContext(
    128, "temp/index.bin", "temp/generated", "temp/episode_data.json", encoder, pca
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
            use_neighbors
        )
        p_net.load_state_dict(torch.load("temp/p_net_retrieval.pt"))
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

    # Compare agent that uses retrieval with removed retrieval
    if args.compare_retrieval:
        test_episodes = 100
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
        
        p_net_retrieval = PolicyNet(
            torch.Size(spatial_obs_space.shape),
            stats_obs_space.shape[0],
            int(act_space.n),
            torch.Size(neighbor_spatial_obs_space.shape),
            neighbor_scalar_obs_space.shape[1],
            True
        )
        p_net_retrieval.load_state_dict(torch.load("temp/retrieval_vs_no_retrieval/p_net_retrieval_yes_200.pt"))

        p_net_no_retrieval = PolicyNet(
            torch.Size(spatial_obs_space.shape),
            stats_obs_space.shape[0],
            int(act_space.n),
            torch.Size(neighbor_spatial_obs_space.shape),
            neighbor_scalar_obs_space.shape[1],
            False
        )
        p_net_no_retrieval.load_state_dict(torch.load("temp/retrieval_vs_no_retrieval/p_net_retrieval_yes_200.pt"))

        test_env = TimeLimit(
            RetrievalMFEnv(
                retrieval_ctx,
                top_k,
                max_skip_frames=max_skip_frames,
                num_frames=num_frames,
            ),
            100,
        )
        with torch.no_grad():
            retrieval_wins = 0
            no_retrieval_wins = 0
            for _ in tqdm(range(test_episodes)):
                (obs_1_, obs_2_, n_obs_1_, n_obs_2_), _ = test_env.reset()
                eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
                eval_n_obs_1 = torch.from_numpy(np.array(n_obs_1_)).float()
                eval_n_obs_2 = torch.from_numpy(np.array(n_obs_2_)).float()
                while True:
                    bot_obs_1, bot_obs_2, bot_n_obs_1, bot_n_obs_2 = test_env.bot_obs()
                    bot_action_probs = p_net_retrieval(
                        torch.from_numpy(bot_obs_1).unsqueeze(0).float(),
                        torch.from_numpy(bot_obs_2).unsqueeze(0).float(),
                        torch.from_numpy(bot_n_obs_1).unsqueeze(0).float(),
                        torch.from_numpy(bot_n_obs_2).unsqueeze(0).float(),
                    ).squeeze()
                    bot_action = Categorical(logits=bot_action_probs).sample().numpy()
                    test_env.bot_step(bot_action)

                    action_probs = p_net_no_retrieval(
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

                    eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                    eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
                    eval_n_obs_1 = torch.from_numpy(np.array(n_obs_1_)).float()
                    eval_n_obs_2 = torch.from_numpy(np.array(n_obs_2_)).float()
                    if eval_done or eval_trunc:
                        if eval_done:
                            if not eval_info["player_won"]:
                                retrieval_wins += 1
                            else:
                                no_retrieval_wins += 1
                        break
        print(f"Retrieval win pct: {(retrieval_wins / test_episodes) * 100}%, No retrieval win pct: {(no_retrieval_wins / test_episodes) * 100}%")
        quit()

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
        use_neighbors
    )
    p_net = PolicyNet(
        torch.Size(spatial_obs_space.shape),
        stats_obs_space.shape[0],
        int(act_space.n),
        torch.Size(neighbor_spatial_obs_space.shape),
        neighbor_scalar_obs_space.shape[1],
        use_neighbors
    )
    if args.resume:
        p_net.load_state_dict(torch.load("temp/p_net_retrieval.pt"))
        v_net.load_state_dict(torch.load("temp/v_net_retrieval.pt"))
    v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
    p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)

    bot_net_path = "temp/training/bot_p_net.ptc"

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

    # Initialize rollout context
    sample_input = obs_space.sample()
    traced = torch.jit.trace(
        p_net,
        (
            torch.from_numpy(sample_input[0]).unsqueeze(0),
            torch.from_numpy(sample_input[1]).unsqueeze(0),
            torch.from_numpy(sample_input[2]).unsqueeze(0),
            torch.from_numpy(sample_input[3]).unsqueeze(0),
        ),
    )
    p_net_path = "temp/training/p_net.ptc"
    traced.save(p_net_path)
    rollout_context = RolloutContext(
        num_envs,
        num_workers,
        train_steps,
        max_skip_frames,
        num_frames,
        time_limit,
        p_net_path,
        top_k,
        start_elo,
        True,
    )
    del retrieval_ctx, test_env, encoder, pca

    for step in tqdm(range(iterations), position=0):
        percent_done = step / iterations
        rollout_context.set_expl_reward_amount(1.0 - percent_done)

        # Collect experience
        print("Performing rollouts...", end="")
        curr_time = time.time()
        traced = torch.jit.trace(
            p_net,
            (
                torch.from_numpy(sample_input[0]).unsqueeze(0),
                torch.from_numpy(sample_input[1]).unsqueeze(0),
                torch.from_numpy(sample_input[2]).unsqueeze(0),
                torch.from_numpy(sample_input[3]).unsqueeze(0),
            ),
        )
        traced.save(p_net_path)
        try:
            (
                (obs_1_buf, obs_2_buf, obs_3_buf, obs_4_buf),
                act_buf,
                act_probs_buf,
                reward_buf,
                done_buf,
                trunc_buf,
                avg_entropy,
            ) = rollout_context.rollout(p_net_path)
        except KeyboardInterrupt:
            quit()
        except Exception as e:
            print(e)
            continue

        avg_train_reward = reward_buf.mean().item()

        buffer_spatial.states.copy_(obs_1_buf)
        buffer_stats.states.copy_(obs_2_buf)
        buffer_n_spatial.states.copy_(obs_3_buf)
        buffer_n_scalar.states.copy_(obs_4_buf)
        buffer_spatial.actions.copy_(act_buf)
        buffer_spatial.action_probs.copy_(act_probs_buf)
        buffer_spatial.rewards.copy_(reward_buf)
        buffer_spatial.dones.copy_(done_buf)
        buffer_spatial.truncs.copy_(trunc_buf)
        del (
            obs_1_buf,
            obs_2_buf,
            obs_3_buf,
            obs_4_buf,
            act_buf,
            act_probs_buf,
            reward_buf,
            done_buf,
            trunc_buf,
        )
        print(f" took {time.time() - curr_time} seconds.")

        # Train
        total_p_loss, total_v_loss, kl_div = train_ppo(
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
            entropy_coeff=entropy_coeff,
        )
        buffer_spatial.clear()
        buffer_stats.clear()
        buffer_n_spatial.clear()
        buffer_n_scalar.clear()

        # Evaluate the network's performance.
        if (step + 1) % eval_every == 0:
            print("Performing evaluation...", end="")
            curr_time = time.time()
            rollout_context.perform_eval(eval_steps, max_eval_steps, elo_k)
            print(f" took {time.time() - curr_time} seconds.")

        wandb.log(
            {
                "current_elo": rollout_context.current_elo(),
                "avg_v_loss": total_v_loss / train_iters,
                "avg_p_loss": total_p_loss / train_iters,
                "kl_div": kl_div,
                "entropy": avg_entropy,
                "avg_train_reward": avg_train_reward,
            }
        )

        # Update bot nets
        if (step + 1) % bot_update == 0:
            traced = torch.jit.trace(
                p_net,
                (
                    torch.from_numpy(sample_input[0]).unsqueeze(0),
                    torch.from_numpy(sample_input[1]).unsqueeze(0),
                    torch.from_numpy(sample_input[2]).unsqueeze(0),
                    torch.from_numpy(sample_input[3]).unsqueeze(0),
                ),
            )
            traced.save(bot_net_path)
            bot_data = rollout_context.bot_data()
            if len(bot_data) < max_bots:
                rollout_context.push_bot(bot_net_path)
            else:
                # Replace bot with lowest ELO if that ELO is less than the current ELO
                next_bot_index = 0
                current_elo = rollout_context.current_elo()
                lowest_elo = current_elo
                for bot_index, data in enumerate(bot_data):
                    if data.elo < lowest_elo:
                        next_bot_index = bot_index
                if current_elo > lowest_elo:
                    rollout_context.insert_bot(bot_net_path, next_bot_index)
            del bot_data

        torch.save(p_net.state_dict(), "temp/p_net_retrieval.pt")
        torch.save(v_net.state_dict(), "temp/v_net_retrieval.pt")
