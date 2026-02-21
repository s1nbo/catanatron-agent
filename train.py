import sys
import numpy as np
import catanatron.gym
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

def mask_fn(env):
    valid_action_indices = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[valid_action_indices] = True
    return mask

def train(time_steps=10, n_envs=8, model_path=None):
    env_kwargs = {
        "config": {
            "enemies": [
                AlphaBetaPlayer(Color.RED),
                AlphaBetaPlayer(Color.ORANGE),
                AlphaBetaPlayer(Color.WHITE),
                
            ]
        }
    }

    env = make_vec_env(
        "catanatron/Catanatron-v0",
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        vec_env_cls=SubprocVecEnv,
        
    )
    
    if model_path:
        print(f"Loading existing model from {model_path}...")
        model = MaskablePPO.load(model_path, env=env, device="cuda")
    else:
        print("Creating new model...")
        model = MaskablePPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            device="cuda", 
            batch_size=4096,
            n_steps=4096,
        )

    print(f"Starting Training...")
    model.learn(total_timesteps=time_steps)
    model.save("ppo_catanatron_03")
    print("Model saved.")
    env.close()

if __name__ == "__main__":
    # Example usage: python train.py 1000000 16 [optional_path_to_model]
    ts = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    model_path = sys.argv[3] if len(sys.argv) > 3 else "ppo_catanatron_03.zip"
    train(time_steps=ts, n_envs=n_envs, model_path=model_path)
