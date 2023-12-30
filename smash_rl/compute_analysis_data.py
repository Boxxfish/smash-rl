"""
Computes data for the memory analysis UI.
"""

import json
import pickle
import gymnasium as gym
import numpy as np
from sklearn.decomposition import PCA  # type: ignore
import torch
from tqdm import tqdm
from smash_rl.experiments.train_ppo import PolicyNet
from smash_rl.micro_fighter.env import MFEnv
from gymnasium.wrappers.time_limit import TimeLimit
from torch.distributions import Categorical
import torch.nn as nn

num_episodes = 10
episode_len = 50
num_matches = 10


def gen_keys(
    keys_spatial: list[torch.Tensor],
    keys_stats: list[torch.Tensor],
    shared_net: nn.Module,
    pca: PCA,
) -> np.ndarray:
    with torch.no_grad():
        outputs = shared_net(
            torch.stack(keys_spatial),
            torch.stack(keys_stats),
        ).numpy()
    outputs = pca.transform(outputs)  # Shape: (batch_size, key_dim)
    outputs = outputs / np.sqrt((outputs**2).sum(1, keepdims=True))
    return outputs


if __name__ == "__main__":
    test_env = TimeLimit(MFEnv(max_skip_frames=2, num_frames=4), episode_len)
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
    with open("temp/pca.pkl", "rb") as rfile:
        pca = pickle.load(rfile)
    shared_net = p_net.shared

    # Generate data for all episodes
    print("Generating data..")
    all_keys = []
    all_visuals = []
    all_indices = []
    all_steps = []
    with torch.no_grad():
        for ep_idx in tqdm(range(num_episodes)):
            (obs_1_, obs_2_), _ = test_env.reset()
            eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
            eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()

            ep_visuals = []
            ep_keys_spatial = []
            ep_keys_scalar = []

            while True:
                # Collect frame data
                visual: np.ndarray = (
                    (
                        np.flip(
                            np.concatenate(
                                [
                                    obs_1_[0][[0, 2, 3]],
                                    np.ones([1] + list(obs_1_[0][0].shape)),
                                ],
                                0,
                            ).transpose(1, 2, 0),
                            0,
                        )
                        * 255.0
                    )
                    .astype(int)
                    .flatten()
                )
                ep_visuals.append(visual.tolist())
                ep_keys_spatial.append(eval_obs_1)
                ep_keys_scalar.append(eval_obs_2)

                # Sample data
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

                eval_obs_1 = torch.from_numpy(np.array(obs_1_)).float()
                eval_obs_2 = torch.from_numpy(np.array(obs_2_)).float()
                if eval_done or eval_trunc:
                    break

            ep_keys = gen_keys(
                ep_keys_spatial, ep_keys_scalar, shared_net, pca
            ).tolist()
            all_visuals.append(ep_visuals)
            all_keys += ep_keys
            all_indices += [ep_idx] * len(ep_keys)
            all_steps += list(range(len(ep_keys)))

    # Save episode data
    print("Saving data...")
    all_keys_arr = np.array(all_keys)
    scores = (all_keys_arr @ all_keys_arr.T) - (
        np.identity(all_keys_arr.shape[0]) * 999
    )
    score_idx = 0
    for ep_idx, (ep_visuals, ep_keys) in tqdm(enumerate(zip(all_visuals, all_keys))):
        matches = []
        for vis_idx in range(len(ep_visuals)):
            frame_scores = scores[score_idx] - (np.array(all_indices) == ep_idx) * 999
            frame_matches = []
            for _ in range(num_matches):
                best_idx = np.argmax(frame_scores)
                score = frame_scores[best_idx]
                episode = all_indices[best_idx]
                start = all_steps[best_idx]
                frame_match = {
                    "score": score,
                    "episode": episode,
                    "start": start,
                }
                frame_matches.append(frame_match)
                frame_scores[best_idx] = -np.inf
            score_idx += 1
            matches.append(frame_matches)

        ep_data = {
            "visuals": ep_visuals,
            "matches": matches,
        }
        with open(f"temp/analyzer/episodes/{ep_idx}.json", "w") as f:
            json.dump(ep_data, f)
