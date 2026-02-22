import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env

# Local imports
from league import League
from self_play_env import SelfPlayEnv

def mask_fn(env):
    valid_action_indices = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[valid_action_indices] = True
    return mask

class LeagueCallback(BaseCallback):
    def __init__(self, league, check_freq: int, model_dir: str, verbose=1):
        super().__init__(verbose)
        self.league = league
        self.check_freq = check_freq
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.generation = 1

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Save current model
            model_name = f"gen_{self.generation}"
            model_path = os.path.join(self.model_dir, f"{model_name}.zip")
            self.model.save(model_path)
            
            if self.verbose > 0:
                print(f"Generation {self.generation} saved to {model_path}")
            
            self.league.add_player(model_name, "ppo", model_path)
            self.generation += 1
            
        return True

def train_selfplay():
    league = League()
    
    # Create env
    n_envs = 32
    
    env = make_vec_env(
        SelfPlayEnv,
        n_envs=n_envs,
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        vec_env_cls=SubprocVecEnv,
    )
    
    # Tuned hyperparameters from Optuna optimization (Hardcoded)
    learning_rate = 0.0009674608384885506
    n_steps = 1024
    batch_size = 512
    ent_coef = 3.641487559642055e-05
    gamma = 0.9996030561768017
    gae_lambda = 0.9998258433696674
    clip_range = 0.38705285305039916
    
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="runs/selfplay",
        device="cuda",
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range
    )
    
    callback = LeagueCallback(league, check_freq=5_000_000, model_dir="league_models")
    
    print("Starting Self-Play Training with tuned parameters")
    model.learn(total_timesteps=10_000_000, callback=callback)
    
    model.save("final_selfplay_model")
    env.close()

if __name__ == "__main__":
    train_selfplay()
