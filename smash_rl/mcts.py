from typing import Optional
import numpy as np
from smash_rl_rust import GameState
import torch
from smash_rl.micro_fighter.env import MFEnv
from torch import nn


class MCTSNode:
    """
    Stores a Monte Carlo Tree Search node.
    """

    def __init__(self, value: float, num_actions: int, discount: float):
        self.total_return = value
        self.visited = 1
        self.discount = discount
        self.num_actions = num_actions
        self.children: Optional[list[MCTSNode]] = None

    def simulate(self, q_net: nn.Module, env: MFEnv):
        """
        Runs simulate step on this node, expanding children as needed.
        """
        player_obs = torch.from_numpy(env.player_obs()).unsqueeze(0)
        player_q_vals = q_net(player_obs).squeeze(0)
        # TODO: Implement PUCT score for action selection
        action = int(torch.argmax(player_q_vals, 0).item())
        opp_obs = torch.from_numpy(env.bot_obs()).unsqueeze(0)
        opp_q_vals = q_net(opp_obs).squeeze(0)
        opp_action = int(torch.argmax(opp_q_vals, 0).item())
        env.bot_step(opp_action)
        _, reward, done, _, _ = env.step(action)

        subsequent_return = 0.0
        if not done:
            if not self.children:
                # Expand child nodes
                self.children = []
                for i in range(self.num_actions):
                    self.children.append(MCTSNode(player_q_vals[i].item(), self.num_actions, self.discount))
            else:
                # Expand child node
                self.children[action].simulate(q_net, env)
            subsequent_return = self.children[action].total_return / self.children[action].visited
            

        self.total_return += reward + self.discount * subsequent_return
        self.visited += 1

def run_mcts(
    q_net: nn.Module,
    env: MFEnv,
    initial_state: GameState,
    rollouts: int,
    discount: float,
    num_actions: int,
) -> int:
    """
    Runs MCTS on the provided env. This method assumes MFEnv is used.
    Returns the action with the highest reward.
    """
    # Root node is special case, we'll have the first expansion set its actual value
    root = MCTSNode(0.0, num_actions, discount)
    root.visited = 0
    for _ in range(rollouts):
        env.load_state(initial_state)
        root.simulate(q_net, env)
    assert root.children
    best_action = max(enumerate(root.children), key=lambda x: x[1].total_return / x[1].visited)[0]
    return best_action
