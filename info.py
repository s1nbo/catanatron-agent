'''
from sb3_contrib.common.wrappers import ActionMasker
import numpy as np

def mask_fn(env):
    # gym.make usually wraps the env in other wrappers (like TimeLimit), 
    # so we use .unwrapped to access the base CatanatronEnv methods
    valid_action_indices = env.unwrapped.get_valid_actions()
    
    # Create a boolean mask of valid actions
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[valid_action_indices] = True
    return mask

def train(time_steps=10000, n_envs=8):
    # ...
    env = make_vec_env(
        "catanatron/Catanatron-v0",
        n_envs=n_envs,
        wrapper_class=ActionMasker,  # <--- Helper from sb3_contrib
        wrapper_kwargs={"action_mask_fn": mask_fn}, # <--- Pass our function here
        vec_env_cls=SubprocVecEnv
    )
    # ...
'''

