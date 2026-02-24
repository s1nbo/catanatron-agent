import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback

# Local imports
from league import League
from self_play_env import SelfPlayEnv

def mask_fn(env):
    valid_action_indices = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[valid_action_indices] = True
    return mask

class LeagueCallback(BaseCallback):
    def __init__(self, league, check_freq: int, model_dir: str, run_name: str = "default", verbose=1, max_league_size=32):
        super().__init__(verbose)
        self.league = league
        self.check_freq = check_freq
        self.model_dir = model_dir
        self.run_name = run_name
        self.max_league_size = max_league_size
        os.makedirs(self.model_dir, exist_ok=True)

        if verbose > 0:
            print(f"LeagueCallback initialized. Check freq: {self.check_freq} steps (n_calls).")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Save current model
            model_name = f"{self.run_name}_{self.num_timesteps}"
            model_path = os.path.join(self.model_dir, f"{model_name}.zip")
            self.model.save(model_path)

            if self.verbose > 0:
                print(f"Saved: {model_path} (Global Timesteps: {self.num_timesteps})")

            # Inherit the current training agent's earned ELO so the snapshot
            # starts from an honest prior rather than the (potentially inflated) league mean.
            training_agent_data = self.league.players.get("current_training_agent")
            inherited_elo = training_agent_data["elo"] if training_agent_data else None

            self.league.add_player(model_name, "ppo", model_path, initial_elo=inherited_elo)

            # Prune old models if league gets too big
            if self.max_league_size > 0:
                self.league.prune_league(self.max_league_size)

            # --- wandb: log generation + league snapshot ---
            league_elos = {
                name: data["elo"]
                for name, data in self.league.players.items()
            }
            ppo_elos = [
                data["elo"]
                for data in self.league.players.values()
                if data.get("type") == "ppo"
            ]
            log_data = {
                "league/training_agent_elo": inherited_elo or 0,
                "league/num_agents": len(self.league.players),
                "league/ppo_mean_elo": float(np.mean(ppo_elos)) if ppo_elos else 0,
                "league/ppo_max_elo": float(np.max(ppo_elos)) if ppo_elos else 0,
            }
            # Log every individual agent's ELO under league/elo/<name>
            for name, elo in league_elos.items():
                log_data[f"league/elo/{name}"] = elo
            wandb.log(log_data, step=self.num_timesteps)

        # Log training-agent ELO every step so it trends smoothly
        training_data = self.league.players.get("current_training_agent")
        if training_data:
            wandb.log(
                {"league/training_agent_elo": training_data["elo"]},
                step=self.num_timesteps,
            )

        return True



def train_selfplay(total_timesteps, check_freq, n_envs, load_path=None, run_name="default"):
    wandb.init(
        project="catanatron-selfplay",
        name=run_name,
        config={
            "total_timesteps": total_timesteps,
            "check_freq": check_freq,
            "n_envs": n_envs,
            "load_path": load_path,
            "run_name": run_name,
        },
        sync_tensorboard=True,   # auto-sync SB3 TensorBoard scalars
        resume="allow",
    )

    league = League()
    max_league_size = 32
    
    # Create env
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

    if load_path and os.path.exists(load_path):
        print(f"Loading existing model from {load_path}...")
        # Note: We need to pass tensorboard_log etc. if we want to continue logging properly
        # However, .load() creates a new object. We should update its attributes if needed.
        # But 'learning_rate' etc are usually baked in.
        # But Stable Baselines3 saves hyperparams. MaskablePPO.load should work fine.
        model = MaskablePPO.load(
            load_path, 
            env=env,
            device="cuda",
            tensorboard_log="runs/selfplay",
            verbose=1,
            custom_objects={
                "learning_rate": learning_rate,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "ent_coef": ent_coef,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range
            }
        )
    else:
        print("Creating new model...")
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
    
    print(f"Starting Self-Play Training for {total_timesteps} steps...")

    # Convert check_freq (timesteps) to n_calls (timesteps // n_envs)
    check_freq_calls = max(1, check_freq // n_envs)
    print(f"Update Frequency: Every {check_freq} timesteps ({check_freq_calls} callback calls)")

    league_cb = LeagueCallback(
        league, check_freq_calls, "league_models",
        run_name=run_name, max_league_size=max_league_size,
    )
    wandb_cb = WandbCallback(
        gradient_save_freq=check_freq_calls,
        verbose=0,
    )
    callback = CallbackList([league_cb, wandb_cb])

    reset_timesteps = load_path is None

    model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=reset_timesteps)

    model.save(f"final_model_{run_name}")
    env.close()
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000_000 , help="Total training steps")
    parser.add_argument("--freq", type=int, default=5_000_000, help="League update frequency")
    parser.add_argument("--envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--load", type=str, default="league_models/v2_gen_9.zip", help="Path to model .zip to resume from")
    parser.add_argument("--name", type=str, default="v3", help="Unique name for this training run (e.g. 'run_A', 'v1')")
    args = parser.parse_args()
    
    train_selfplay(args.steps, args.freq, args.envs, args.load, args.name)
