
import gymnasium as gym
from self_play_env import SelfPlayEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.wrappers import ActionMasker
import numpy as np

def mask_fn(env):
    valid_action_indices = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[valid_action_indices] = True
    return mask

def test_environment():
    print("Initializing SelfPlayEnv...")
    env = SelfPlayEnv()
    obs, info = env.reset()
    print("Environment reset.")
    
    done = False
    steps = 0
    while not done and steps < 1000:
        # Random valid action
        valid_actions = env.get_valid_actions()
        action = np.random.choice(valid_actions)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        
        if done:
            print(f"Game finished in {steps} steps.")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            print(f"Info keys: {info.keys()}")
            print(f"Winning color in info: {info.get('winning_color')}")
            if hasattr(env, "game"):
                wc = env.game.winning_color
                if callable(wc):
                    print(f"Game winning color (called): {wc()}")
                else:
                    print(f"Game winning color (property): {wc}")
            else:
                print("env.game not accessible")

if __name__ == "__main__":
    test_environment()
