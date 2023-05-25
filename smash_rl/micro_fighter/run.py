"""
Runs the environment in human mode.
"""
from smash_rl_rust import MicroFighterEnv

def main():
    env = MicroFighterEnv()
    env.run()

if __name__ == "__main__":
    main()