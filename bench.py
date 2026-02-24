import os
import time

# 1. STOP THREAD THRASHING BEFORE IMPORTING ANYTHING ELSE
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Replace this with your actual Catanatron environment creation
def make_env():
    def _init():
        
        from self_play_env import SelfPlayEnv
        env = SelfPlayEnv()
        return env
    return _init

def run_benchmark():
    # Test these environment counts (staying under your 24 logical cores)
    env_counts_to_test = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80]
    total_steps = 5000 # Enough steps to get past the startup spike
    
    results = {}

    print("--- Starting Catanatron Environment Benchmark ---")
    
    for n_envs in env_counts_to_test:
        print(f"\nTesting with {n_envs} parallel environments...")
        
        # Create environments
        envs = SubprocVecEnv([make_env() for _ in range(n_envs)])
        
        # Initialize a lightweight dummy model
        model = PPO("MlpPolicy", envs, n_steps=128, verbose=0)
        
        # Time the learning process
        start_time = time.time()
        model.learn(total_timesteps=total_steps)
        end_time = time.time()
        
        # Calculate performance
        duration = end_time - start_time
        fps = total_steps / duration
        results[n_envs] = fps
        
        print(f"Result: {fps:.2f} Steps Per Second (Total time: {duration:.2f}s)")
        
        # Clean up memory before the next loop!
        envs.close()
        del model
        del envs

    # Print the final verdict
    print("\n=== FINAL RESULTS ===")
    for n, fps in results.items():
        print(f"{n} Envs: {fps:.0f} FPS")
        
    best_n = max(results, key=results.get)
    print(f"\n🏆 OPTIMAL SETTING: Set n_envs = {best_n} (Speed: {results[best_n]:.0f} FPS)")

if __name__ == "__main__":
    run_benchmark()
