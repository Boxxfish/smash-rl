"""
Tests the micro fighter environment.
"""

from smash_rl.micro_fighter.env import MFEnv

env = MFEnv(render_mode="human", view_channels=(0, 2, 3), max_skip_frames=4)
env.reset()
action_space = env.action_space
while True:
    env.bot_step(action_space.sample())
    _obs, _reward, terminated, _truncated, _info = env.step(action_space.sample())
    env.render()
    if terminated:
        env.reset()