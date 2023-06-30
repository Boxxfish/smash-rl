from typing import Optional
import numpy as np
from smash_rl_rust import GameState
import torch
from smash_rl.micro_fighter.env import MFEnv
from torch import nn


class MCTSNode:
    """
    Stores a Monte Carlo Tree Search node, containing either the guess for this
    node's value, or the neighbor's value.
    The children of a node correspond to actions the opponent performs.
    """

    def __init__(self, value: float, num_actions: int, is_player: bool):
        self.value = value
        self.num_actions = num_actions
        self.is_player = is_player
        self.children: Optional[list[MCTSNode]] = None

    def simulate(self, q_net: nn.Module, env: MFEnv):
        """
        Runs simulate step on this node, expanding children as needed.
        """
        if not self.children:
            # Expand child nodes
            if self.is_player:
                obs = torch.from_numpy(env.player_obs()).unsqueeze(0)
            else:
                obs = torch.from_numpy(env.bot_obs()).unsqueeze(0)
            q_vals = q_net(obs).squeeze(0)
            self.children = []
            for i in range(self.num_actions):
                self.children.append(MCTSNode(q_vals[i].item(), self.num_actions, not self.is_player))
            
            # Update current value with children two laye, if they exist
        

def run_mcts(
    q_net: nn.Module,
    env: MFEnv,
    initial_obs: np.ndarray,
    initial_state: GameState,
    rollouts: int,
    discount: float,
    num_actions: int,
) -> int:
    """
    Runs MCTS on the provided env. This method assumes MFEnv is used.
    Returns the method with the highest reward.
    """
    obs_ = initial_obs
    obs = torch.from_numpy(obs_).unsqueeze(0)
    q_vals = q_net(obs).squeeze(0)
    root = MCTSNode(0.0, num_actions)
    root.children = []
    for i in range(num_actions):
        root.children.append(MCTSNode(q_vals[i].item(), num_actions))
    for _ in range(rollouts):
        curr_node = root
        while curr_node.children:
            # TODO: Incorporate UCT
            curr_node = max(enumerate(curr_node.children), key=lambda x: x[1].value)[1]
    return 0
