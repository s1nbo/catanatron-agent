import os
import numpy as np
import optuna
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

def get_optuna_params():
    print("Loading best hyperparameters from Optuna db...")
    try:
        # Load the study, assuming it exists in the current directory
        study = optuna.load_study(
            study_name="catanatron_ppo_optimization", 
            storage="sqlite:///catanatron_optuna.db"
        )
        params = study.best_params
        print(f"Best params found: {params}")
        return params
    except Exception as e:
        print(f"Could not load optuna study: {e}. Using defaults.")
        # If loading fails, return empty dict to fallback to defaults
        return {}

def train_selfplay():
    league = League()
    
    # Create env
    # Use 64 processes or less depending on machine
    n_envs = 64
    
    env = make_vec_env(
        SelfPlayEnv,
        n_envs=n_envs,
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        vec_env_cls=SubprocVecEnv,
    )
    
    # Load parameters
    params = get_optuna_params()
    
    # Tuned hyperparameters from Optuna optimization
    learning_rate = params.get("learning_rate", 0.0009674608384885506)
    n_steps = params.get("n_steps", 1024)
    batch_size = params.get("batch_size", 512)
    ent_coef = params.get("ent_coef",3.641487559642055e-05)
    gamma = params.get("gamma",  0.9996030561768017)
    gae_lambda = params.get("gae_lambda", 0.9998258433696674)
    clip_range = params.get("clip_range",  0.38705285305039916)
    
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
    
    callback = LeagueCallback(league, check_freq=100000, model_dir="league_models")
    
    print("Starting Self-Play Training with parameters:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  N Steps: {n_steps}")
    print(f"  Batch Size: {batch_size}")
    
    model.learn(total_timesteps=10_000_000, callback=callback)
    
    model.save("final_selfplay_model")
    env.close()

if __name__ == "__main__":
    train_selfplay()
