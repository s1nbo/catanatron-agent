import gymnasium
import catanatron.gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import os
import sys

def mask_fn(env):
    valid_action_indices = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[valid_action_indices] = True
    return mask

def train(time_steps=10000, n_envs=8):
    print(f"Initializing {n_envs} Parallel Environments (SubprocVecEnv)...")
    
    # Create vectorized environment to run multiple games in parallel
    # This feeds data to the GPU faster, significantly speeding up training.
    # We apply ActionMasker to each individual environment.
    env = make_vec_env(
        "catanatron/Catanatron-v0",
        n_envs=n_envs,
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        vec_env_cls=SubprocVecEnv
    )
    
    print("Initializing Agent...")
    # device="auto" puts the neural network on GPU if available
    model = MaskablePPO("MlpPolicy", env, verbose=1, device="auto", batch_size=2048)

    print(f"Starting Training ({time_steps} steps)...")
    # For good performance, you'd want millions.
    model.learn(total_timesteps=time_steps)
    
    print("Saving Model...")
    model.save("ppo_catanatron-2")
    print("Model saved to ppo_catanatron-2.zip")
    env.close()

if __name__ == "__main__":
    # read command line args for time steps,
    time_steps = 10000
    n_envs = 8
    
    if len(sys.argv) > 1:
        time_steps = int(sys.argv[1])
    
    if len(sys.argv) > 2:
        n_envs = int(sys.argv[2])
        
    train(time_steps, n_envs)