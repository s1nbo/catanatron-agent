import optuna
import wandb
import numpy as np
import warnings
# Filter specific warning from gymnasium about render_mode
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.envs.registration")

import catanatron.gym
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

class TrialEvalCallback(BaseCallback):
    """Callback used for evaluating and reporting a trial."""
    def __init__(self, trial: optuna.Trial, verbose=0):
        super().__init__(verbose)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.n_calls % 10000 == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                self.trial.report(mean_reward, self.n_calls)
                
                # Check for pruning
                if self.trial.should_prune():
                    self.is_pruned = True
                    return False
        return True

def mask_fn(env):
    valid_action_indices = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[valid_action_indices] = True
    return mask

def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)

    # Initialize wandb run for this trial
    run = wandb.init(
        project="catanatron-tuning",
        config=trial.params,
        reinit=True,
        sync_tensorboard=True,
        monitor_gym=False,  # Explicitly disable gym monitoring
    )
    
    # Ensure batch_size is a factor of n_steps * n_envs to avoid errors
    # SB3 requires n_steps * n_envs to be greater than batch_size
    # Default n_steps is usually per-env.
    n_envs = 64 
     
    env_kwargs = {
        "config": {
            "enemies": [
                RandomPlayer(Color.RED),
                RandomPlayer(Color.ORANGE),
                RandomPlayer(Color.WHITE),

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

    try:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            tensorboard_log=f"runs/{run.id}",
            device="cuda"
        )

        callback = TrialEvalCallback(trial)
        
        # Train for a limited number of steps per trial to save time
        total_timesteps = 100_000
        model.learn(total_timesteps=total_timesteps, callback=callback)

        if callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        # Evaluate performance using the last 100 episodes
        mean_reward = 0
        if len(model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in model.ep_info_buffer])
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        mean_reward = -float('inf') 
        
    finally:
        env.close()
        run.finish()

    return mean_reward

if __name__ == "__main__":
    # Create the study
    sampler = TPESampler(n_startup_trials=5)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10000)
    
    study = optuna.create_study(
        study_name="catanatron_ppo_optimization",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///catanatron_optuna.db",
        load_if_exists=True
    )

    print("Starting optimization...")
    study.optimize(objective, n_trials=1000, n_jobs=1) # n_jobs=1 because SubprocVecEnv spawns processes

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
