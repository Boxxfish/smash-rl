"""
Tests the micro fighter environment.
"""

from smash_rl.micro_fighter.env import MFEnv

env = MFEnv(render_mode="human", view_channels=(0, 2, 4))
# env.reset()
action_space = env.action_space
while True:
    _obs, _reward, _truncated, terminated, _info = env.step(action_space.sample())
    env.render()
    if terminated:
        env.reset()