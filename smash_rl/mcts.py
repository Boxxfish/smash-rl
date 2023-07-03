import math
from typing import Optional
import numpy as np
from smash_rl_rust import GameState
import torch
from smash_rl.micro_fighter.env import MFEnv, EnvState
from torch import nn

# Constant for MCTS.
# Higher values encourage exploration.
C_PUCT = 4.0

class MCTSNode:
    """
    Stores a Monte Carlo Tree Search node.
    """

    def __init__(self, value: float, num_actions: int, discount: float):
        self.total_return = value
        self.visited = 0
        self.discount = discount
        self.num_actions = num_actions
        self.children: Optional[list[MCTSNode]] = None

    def simulate(self, q_net: nn.Module, env: MFEnv):
        """
        Runs simulate step on this node, expanding children as needed.
        """
        if not self.children:
            # Expand child nodes when first expanding
            player_obs = torch.from_numpy(env.player_obs()).float().unsqueeze(0)
            player_q_vals = q_net(player_obs).squeeze(0)
            self.children = []
            for i in range(self.num_actions):
                self.children.append(MCTSNode(player_q_vals[i].item(), self.num_actions, self.discount))
        
        # Choose action and simulate next step
        total_visited_sqrt = math.sqrt(sum([x.visited for x in self.children]))
        action = max(enumerate(self.children), key=lambda x: x[1].puct(total_visited_sqrt))[0]
        opp_obs = torch.from_numpy(env.bot_obs()).float().unsqueeze(0)
        opp_q_vals = q_net(opp_obs).squeeze(0)
        opp_action = int(torch.argmax(opp_q_vals, 0).item())
        env.bot_step(opp_action)
        print(action)
        _, reward, done, _, _ = env.step(action)
        env.render()
        
        if done:
            subsequent_return = 0.0
        elif self.children[action].visited == 0:
            # Don't recurse
            subsequent_return = self.children[action].avg_value()
            self.children[action].visited += 1
        else:
            # Expand child node
            self.children[action].simulate(q_net, env)
            subsequent_return = self.children[action].avg_value()
            
        self.total_return += reward + self.discount * subsequent_return
        self.visited += 1

    def avg_value(self) -> float:
        """
        Returns the average value experienced by this node.
        """
        if self.visited == 0:
            return self.total_return
        return self.total_return / self.visited

    def puct(self, total_visited_sqrt: float) -> float:
        """
        Returns the PUCT score.
        Currently assumes uniform probablities.
        """
        q = self.avg_value()
        u = C_PUCT * (1 / self.num_actions) * total_visited_sqrt / (1 + self.visited)
        return q + u

def run_mcts(
    q_net: nn.Module,
    env: MFEnv,
    initial_state: EnvState,
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
    print("Finished rollouts")
    assert root.children
    best_action = max(enumerate(root.children), key=lambda x: x[1].avg_value())[0]
    return best_action
