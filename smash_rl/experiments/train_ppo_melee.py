"""
Trains an agent with PPO.
"""
from argparse import ArgumentParser
import copy
from functools import reduce
import json
import pickle
import random
import time
from typing import Any, Mapping, Union
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA  # type: ignore
from horapy import HNSWIndex # type: ignore

import torch
import torch.nn as nn
import wandb
from gymnasium.wrappers.time_limit import TimeLimit
from torch.distributions import Categorical
from tqdm import tqdm
from gymnasium.wrappers.normalize import NormalizeReward

from smash_rl.algorithms.ppo_multi import train_ppo
from smash_rl.algorithms.rollout_buffer import RolloutBuffer, StateRolloutBuffer
from smash_rl.conf import entity
from smash_rl.melee.env import MeleeEnv
from smash_rl.utils import init_orthogonal

_: Any

# Hyperparameters
num_envs = 4  # Number of environments to step through at once during sampling.
train_steps = 2048  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs.
iterations = 1000  # Number of sample/train iterations.
train_iters = 4  # Number of passes over the samples collected.
train_batch_size = 256  # Minibatch size while training models.
discount = 0.999  # Discount factor applied to rewards.
lambda_ = 0.95  # Lambda for GAE.
epsilon = 0.2  # Epsilon for importance sample clipping.
v_lr = 0.001  # Learning rate of the value net.
p_lr = 0.0003  # Learning rate of the policy net.
num_frames = 4  # Number of frames in frame stack.
time_limit = 99999  # Time limit before truncation.
bot_update = 10  # Number of iterations before caching the current policy.
max_bots = 4  # Maximum number of bots to store.
start_elo = 1200  # Starting ELO score for each agent.
elo_k = 16  # ELO adjustment constant.
eval_every = 4  # Number of iterations before evaluating.
eval_steps = 4  # Number of eval runs to perform.
max_eval_steps = 500  # Max number of steps to take during each eval run.
entropy_coeff = 0.00003
device = torch.device("cuda")  # Device to use during training.

# Argument parsing
parser = ArgumentParser()
parser.add_argument("--eval", action="store_true")
parser.add_argument("--pca", action="store", default=None)
parser.add_argument("--generate", action="store", default=None)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

PCA_DIM = 3

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


test_env = MeleeEnv(
    console_id=100 if not args.eval else 101, num_frames=num_frames, render_mode="human"
)

# Initialize policy and value networks
obs_space = test_env.observation_space
act_space = test_env.action_space
assert isinstance(obs_space, gym.spaces.Tuple)
assert isinstance(obs_space.spaces[0], gym.spaces.Box)
assert isinstance(obs_space.spaces[1], gym.spaces.Box)
spatial_obs_space = obs_space.spaces[0]
stats_obs_space = obs_space.spaces[1]
assert isinstance(act_space, gym.spaces.Discrete)
v_net = ValueNet(torch.Size(spatial_obs_space.shape), stats_obs_space.shape[0])
p_net = PolicyNet(
    torch.Size(spatial_obs_space.shape), stats_obs_space.shape[0], int(act_space.n)
)
if args.resume:
    v_net.load_state_dict(torch.load("temp/v_net.pt"))
    p_net.load_state_dict(torch.load("temp/p_net.pt"))
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

# If evaluating, load the latest policy
if args.eval:
    eval_done = False
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    assert isinstance(obs_space, gym.spaces.Tuple)
    assert isinstance(obs_space.spaces[0], gym.spaces.Box)
    assert isinstance(obs_space.spaces[1], gym.spaces.Box)
    spatial_obs_space = obs_space.spaces[0]
    stats_obs_space = obs_space.spaces[1]
    assert isinstance(act_space, gym.spaces.Discrete)
    p_net = PolicyNet(
        torch.Size(spatial_obs_space.shape),
        stats_obs_space.shape[0],
        int(act_space.n),
    )
    p_net.load_state_dict(torch.load("temp/p_net.pt"))
    with torch.no_grad():
        (obs_1_, obs_2_), _ = test_env.reset()
        eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
        eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
        while True:
            bot_obs_1, bot_obs_2 = test_env.bot_obs()
            bot_action_probs = p_net(
                torch.from_numpy(bot_obs_1).unsqueeze(0).float(),
                torch.from_numpy(bot_obs_2).unsqueeze(0).float(),
            ).squeeze()
            bot_action = Categorical(probs=bot_action_probs.exp()).sample().numpy()
            test_env.bot_step(bot_action)

            action_probs = p_net(
                eval_obs_1.unsqueeze(0), eval_obs_2.unsqueeze(0)
            ).squeeze()
            action = Categorical(probs=action_probs.exp()).sample().numpy()
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

# If setting up PCA, train a PCA model
if args.pca:
    gen_count = args.pca

    obs_space = test_env.observation_space
    act_space = test_env.action_space
    assert isinstance(obs_space, gym.spaces.Tuple)
    assert isinstance(obs_space.spaces[0], gym.spaces.Box)
    assert isinstance(obs_space.spaces[1], gym.spaces.Box)
    spatial_obs_space = obs_space.spaces[0]
    stats_obs_space = obs_space.spaces[1]
    assert isinstance(act_space, gym.spaces.Discrete)
    p_net = PolicyNet(
        torch.Size(spatial_obs_space.shape),
        stats_obs_space.shape[0],
        int(act_space.n),
    )
    v_net = ValueNet(torch.Size(spatial_obs_space.shape), stats_obs_space.shape[0])
    p_net.load_state_dict(torch.load("temp/p_net.pt"))
    test_env = MeleeEnv(console_id=0)

    # Allocate key and data arrays
    keys_spatial = []
    keys_stats = []
    with torch.no_grad():
        (obs_1_, obs_2_), _ = test_env.reset()
        eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
        eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
        for j in tqdm(range(int(gen_count))):
            # Store info for key
            keys_spatial.append(eval_obs_1.numpy())
            keys_stats.append(eval_obs_2.numpy())
            traj_len = 32
            for i in range(traj_len):
                bot_obs_1, bot_obs_2 = test_env.bot_obs()
                bot_action_probs = p_net(
                    torch.from_numpy(bot_obs_1).unsqueeze(0).float(),
                    torch.from_numpy(bot_obs_2).unsqueeze(0).float(),
                ).squeeze()
                bot_action = Categorical(probs=bot_action_probs.exp()).sample().numpy()
                test_env.bot_step(bot_action)

                action_probs = p_net(
                    eval_obs_1.unsqueeze(0), eval_obs_2.unsqueeze(0)
                ).squeeze()
                action = Categorical(probs=action_probs.exp()).sample().numpy()
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
                    break

    # Fit PCA
    pca_dim = PCA_DIM
    p_pca = IncrementalPCA(n_components=pca_dim)
    p_shared_net = p_net.shared
    v_pca = IncrementalPCA(n_components=pca_dim)
    v_shared_net = v_net.shared
    with torch.no_grad():
        spatial = torch.from_numpy(np.stack(keys_spatial))
        stats = torch.from_numpy(np.stack(keys_stats))
        p_outputs = p_shared_net(spatial, stats)
        v_outputs = v_shared_net(spatial, stats)
    p_outputs = p_outputs.numpy()
    p_pca.fit(p_outputs)
    del p_outputs
    v_outputs = v_outputs.numpy()
    v_pca.fit(v_outputs)
    del v_outputs
    with open("temp/p_pca.pkl", "wb") as file:
        pickle.dump(p_pca, file)
    with open("temp/v_pca.pkl", "wb") as file:
        pickle.dump(v_pca, file)

    quit()

# If generating, save trajectories
if args.generate:
    gen_count = args.generate
    max_per_part = 512

    obs_space = test_env.observation_space
    act_space = test_env.action_space
    assert isinstance(obs_space, gym.spaces.Tuple)
    assert isinstance(obs_space.spaces[0], gym.spaces.Box)
    assert isinstance(obs_space.spaces[1], gym.spaces.Box)
    spatial_obs_space = obs_space.spaces[0]
    stats_obs_space = obs_space.spaces[1]
    assert isinstance(act_space, gym.spaces.Discrete)
    p_net = PolicyNet(
        torch.Size(spatial_obs_space.shape),
        stats_obs_space.shape[0],
        int(act_space.n),
    )
    p_net.load_state_dict(torch.load("temp/p_net.pt"))
    v_net = ValueNet(
        torch.Size(spatial_obs_space.shape),
        stats_obs_space.shape[0]
    )
    v_net.load_state_dict(torch.load("temp/v_net.pt"))
    test_env = MeleeEnv(console_id=0)

    # Index stuff
    pca_dim = PCA_DIM
    p_index = HNSWIndex(pca_dim, "usize")
    v_index = HNSWIndex(pca_dim, "usize")
    with open("temp/p_pca.pkl", "rb") as rfile:
        p_pca = pickle.load(rfile)
    with open("temp/v_pca.pkl", "rb") as rfile:
        v_pca = pickle.load(rfile)
    p_shared_net = p_net.shared
    v_shared_net = v_net.shared

    def gen_keys(
        keys_spatial: list[np.ndarray],
        keys_stats: list[np.ndarray],
        shared_net: nn.Module,
        pca: PCA,
    ) -> np.ndarray:
        with torch.no_grad():
            outputs = shared_net(
                torch.from_numpy(np.stack(keys_spatial)),
                torch.from_numpy(np.stack(keys_stats)),
            ).numpy()
        outputs = pca.transform(outputs)
        return outputs

    # Allocate key and data arrays
    keys_spatial = []
    keys_stats = []
    data_spatial = []
    data_scalar = []
    episode_data: dict[str, Any] = {}
    episode_datas = []
    episode_id = 0
    traj_in_episode = []
    frames_in_episode = []
    part_id = 0
    with torch.no_grad():
        (obs_1_, obs_2_), _ = test_env.reset()
        eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
        eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
        for j in tqdm(range(int(gen_count))):
            traj_in_episode.append(j)
            frames_in_episode.append(test_env.frame)

            # Store info for key
            keys_spatial.append(eval_obs_1.numpy())
            keys_stats.append(eval_obs_2.numpy())
            traj_len = 32
            traj_data_spatial = np.zeros([traj_len] + list(eval_obs_1.shape[1:]))
            traj_data_scalar = np.zeros(
                [traj_len] + [eval_obs_2.shape[0] + int(act_space.n)]
            )

            for i in range(traj_len):
                bot_obs_1, bot_obs_2 = test_env.bot_obs()
                bot_action_probs = p_net(
                    torch.from_numpy(bot_obs_1).unsqueeze(0).float(),
                    torch.from_numpy(bot_obs_2).unsqueeze(0).float(),
                ).squeeze()
                bot_action = Categorical(probs=bot_action_probs.exp()).sample().numpy()
                test_env.bot_step(bot_action)

                action_probs = p_net(
                    eval_obs_1.unsqueeze(0), eval_obs_2.unsqueeze(0)
                ).squeeze()
                action = Categorical(probs=action_probs.exp()).sample().numpy()
                (
                    (obs_1_, obs_2_),
                    reward,
                    eval_done,
                    eval_trunc,
                    eval_info,
                ) = test_env.step(action)

                # Save trajectory data
                traj_data_spatial[i] = obs_1_[
                    0
                ]  # Only save most recent spatial observation in stack
                act_tensor = np.zeros([int(act_space.n)])
                act_tensor[int(action)] = 1
                traj_data_scalar[i] = np.concatenate([obs_2_, act_tensor])
                timestamp = int(time.time())

                test_env.render()
                eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
                if eval_done or eval_trunc:
                    (obs_1_, obs_2_), _ = test_env.reset()
                    eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                    eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()

                    # Save episode data
                    round_result = 0
                    if eval_done:
                        if eval_info["player_won"]:
                            round_result = 1
                        else:
                            round_result = -1
                    else:
                        round_result = 0
                    episode_data["player_won"] = round_result
                    episode_data["traj_in_episode"] = traj_in_episode
                    episode_data["timestamp"] = timestamp
                    episode_data["frames_in_episode"] = frames_in_episode
                    episode_datas.append(episode_data)
                    episode_data = {}
                    traj_in_episode = []
                    frames_in_episode = []
                    episode_id += 1

                    break

            # Add trajectory data
            data_spatial.append(
                traj_data_spatial.reshape(
                    [
                        traj_data_spatial.shape[0] * traj_data_spatial.shape[1],
                        traj_data_spatial.shape[2],
                        traj_data_spatial.shape[3],
                    ]
                )
            )
            data_scalar.append(
                traj_data_scalar.reshape(
                    [traj_data_scalar.shape[0] * traj_data_scalar.shape[1]]
                )
            )

            # If max number of trajectories have been saved, save to disk
            if len(keys_spatial) == max_per_part:
                # np.save(
                #     f"temp/generated/{part_id}_data_spatial.npy",
                #     np.stack(data_spatial),
                # )
                # np.save(
                #     f"temp/generated/{part_id}_data_scalar.npy",
                #     np.stack(data_scalar),
                # )
                p_keys = gen_keys(keys_spatial, keys_stats, p_shared_net, p_pca)
                v_keys = gen_keys(keys_spatial, keys_stats, v_shared_net, v_pca)
                for i in range(len(p_keys)):
                    p_index.add(np.float32(p_keys[i]), part_id * max_per_part + i)
                    v_index.add(np.float32(v_keys[i]), part_id * max_per_part + i)
                np.save(
                    f"temp/generated/{part_id}_p_keys.npy",
                    p_keys,
                )
                np.save(
                    f"temp/generated/{part_id}_v_keys.npy",
                    v_keys,
                )
                p_index.dump("temp/p_index.bin")
                v_index.dump("temp/v_index.bin")

                with open("temp/episode_data.json", "w") as wfile:
                    json.dump(episode_datas, wfile)
                keys_spatial = []
                keys_stats = []
                data_spatial = []
                data_scalar = []
                part_id += 1

        # One final save
        if len(keys_spatial) > 0:
            # np.save(
            #     f"temp/generated/{part_id}_data_spatial.npy", np.stack(data_spatial)
            # )
            # np.save(f"temp/generated/{part_id}_data_scalar.npy", np.stack(data_scalar))
            p_keys = gen_keys(keys_spatial, keys_stats, p_shared_net, p_pca)
            v_keys = gen_keys(keys_spatial, keys_stats, v_shared_net, v_pca)
            for i in range(len(p_keys)):
                p_index.add(np.float32(p_keys[i]), part_id * max_per_part + i)
                v_index.add(np.float32(v_keys[i]), part_id * max_per_part + i)
            np.save(
                f"temp/generated/{part_id}_p_keys.npy",
                p_keys,
            )
            np.save(
                f"temp/generated/{part_id}_v_keys.npy",
                v_keys,
            )
            p_index.dump("temp/p_index.bin")
            v_index.dump("temp/v_index.bin")
            with open("temp/episode_data.json", "w") as wfile:
                json.dump(episode_datas, wfile)

        quit()

env = SyncVectorEnv(
    [
        lambda i=i: TimeLimit(  # type: ignore
            NormalizeReward(MeleeEnv(console_id=i, num_frames=num_frames)),
            time_limit,
        )
        for i in range(num_envs)
    ]
)

wandb.init(
    project="smash-rl",
    entity=entity,
    config={
        "experiment": "melee ppo",
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

# (obs_1_, obs_2_), _ = env.reset()
# obs_1 = torch.from_numpy(obs_1_).float()
# obs_2 = torch.from_numpy(obs_2_).float()
for step in tqdm(range(iterations), position=0):
    percent_done = step / iterations
    # for env_ in env.envs:
    # assert isinstance(env_, TimeLimit)
    # assert isinstance(env_.env, MeleeEnv)
    # env_.env.set_dmg_reward_amount(1.0 - percent_done)

    # Collect experience
    # env.close()
    # time.sleep(1.0)
    # env = SyncVectorEnv(
    #     [
    #         lambda: TimeLimit(
    #             MeleeEnv(console_id=i, num_frames=num_frames),
    #             time_limit,
    #         )
    #         for i in range(num_envs)
    #     ]
    # )
    # time.sleep(1.0)
    for env_ in env.envs:
        assert isinstance(env_, TimeLimit)
        assert isinstance(env_.env, NormalizeReward)
        assert isinstance(env_.env.env, MeleeEnv)
        env_.env.env.reconnect()
    (obs_1_, obs_2_), _ = env.reset()
    obs_1 = torch.from_numpy(obs_1_).float()
    obs_2 = torch.from_numpy(obs_2_).float()
    with torch.no_grad():
        total_entropy = 0.0
        for _ in tqdm(range(train_steps), position=1):
            # Choose bot action
            for env_index, env_ in enumerate(env.envs):
                assert isinstance(env_, TimeLimit)
                assert isinstance(env_.env, NormalizeReward)
                assert isinstance(env_.env.env, MeleeEnv)
                bot_obs_1, bot_obs_2 = env_.env.bot_obs()
                bot_action_probs = bot_nets[bot_p_indices[env_index]](
                    torch.from_numpy(bot_obs_1).float().unsqueeze(0),
                    torch.from_numpy(bot_obs_2).float().unsqueeze(0),
                ).squeeze(0)
                bot_action = Categorical(probs=bot_action_probs.exp()).sample().item()
                env_.env.bot_step(int(bot_action))

            # Choose player action
            action_probs = p_net(obs_1, obs_2)
            action_distr = Categorical(probs=action_probs.exp())
            total_entropy += action_distr.entropy().mean()
            actions = action_distr.sample().numpy()
            (obs_spatial_, obs_stats_), rewards, dones, truncs, _ = env.step(actions)

            # try:
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

            # Change opponent when environment ends
            for env_index, env_ in enumerate(env.envs):
                assert isinstance(env_, TimeLimit)
                assert isinstance(env_.env, NormalizeReward)
                assert isinstance(env_.env.env, MeleeEnv)
                if dones[env_index] or truncs[env_index]:
                    state_dict = random.choice(bot_data)["state_dict"]
                    assert isinstance(state_dict, Mapping)
                    bot_p_indices[env_index] = random.randrange(0, len(bot_data))
                    bot_nets[bot_p_indices[env_index]].load_state_dict(state_dict)

            # except KeyboardInterrupt:
            #     quit()
            # except Exception as e:
            #     print(f"Exception in simulation: {e}. Resetting.")
            #     (obs_1_, obs_2_), _ = env.reset()
            #     obs_1 = torch.from_numpy(obs_1_).float()
            #     obs_2 = torch.from_numpy(obs_2_).float()
        buffer_spatial.insert_final_step(obs_1)
        buffer_stats.insert_final_step(obs_2)

    for env_ in env.envs:
        assert isinstance(env_, TimeLimit)
        assert isinstance(env_.env, NormalizeReward)
        assert isinstance(env_.env.env, MeleeEnv)
        env_.env.disconnect()

    avg_reward = buffer_spatial.rewards.mean().item()

    # Train
    total_p_loss, total_v_loss, kl_div = train_ppo(
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
        entropy_coeff=entropy_coeff,
    )
    buffer_spatial.clear()
    buffer_stats.clear()

    # Evaluate the network's performance.
    if (step + 1) % eval_every == 0:
        eval_done = False
        with torch.no_grad():
            test_env.reconnect()
            (obs_1_, obs_2_), _ = test_env.reset()
            eval_obs_1 = torch.from_numpy(obs_1_).float()
            eval_obs_2 = torch.from_numpy(obs_2_).float()
            for _ in range(eval_steps):
                eval_bot_index = random.randrange(0, len(bot_data))
                b_elo = bot_data[eval_bot_index]["elo"]
                assert isinstance(b_elo, int)
                try:
                    for _ in range(max_eval_steps):
                        bot_obs_1, bot_obs_2 = test_env.bot_obs()
                        bot_action_probs = bot_nets[eval_bot_index](
                            torch.from_numpy(bot_obs_1).unsqueeze(0).float(),
                            torch.from_numpy(bot_obs_2).unsqueeze(0).float(),
                        ).squeeze()
                        bot_action = (
                            Categorical(probs=bot_action_probs.exp()).sample().numpy()
                        )
                        test_env.bot_step(bot_action)

                        action_probs = p_net(
                            eval_obs_1.unsqueeze(0), eval_obs_2.unsqueeze(0)
                        ).squeeze()
                        actions = Categorical(probs=action_probs.exp()).sample().numpy()
                        (
                            (obs_1_, obs_2_),
                            reward,
                            eval_done,
                            eval_trunc,
                            eval_info,
                        ) = test_env.step(actions)

                        eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                        eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
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
                            ea = 1.0 / (1.0 + 10.0 ** ((b_elo - current_elo) / 400.0))
                            eb = 1.0 / (1.0 + 10.0 ** ((current_elo - b_elo) / 400.0))
                            current_elo = current_elo + elo_k * (a - ea)
                            bot_data[eval_bot_index]["elo"] = int(
                                b_elo + elo_k * (b - eb)
                            )
                            (obs_1_, obs_2_), info = test_env.reset()
                            eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                            eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
                            break
                except KeyboardInterrupt:
                    quit()
                except Exception as e:
                    print(f"Exception during evaluation: {e}.")
                    pass
    test_env.disconnect()

    wandb.log(
        {
            "avg_v_loss": total_v_loss / train_iters,
            "avg_p_loss": total_p_loss / train_iters,
            "kl_div": kl_div,
            "entropy": total_entropy / train_steps,
            "avg_train_reward": avg_reward,
            "current_elo": current_elo,
        }
    )

    # Update bot nets
    if (step + 1) % bot_update == 0:
        if len(bot_data) < max_bots:
            bot_data.append({"state_dict": p_net.state_dict(), "elo": int(current_elo)})
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

    torch.save(p_net.state_dict(), "temp/p_net.pt")
    torch.save(v_net.state_dict(), "temp/v_net.pt")
