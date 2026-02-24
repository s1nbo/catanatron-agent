# Self-Play League System

This folder contains a complete self-play league system for training the Catanatron agent. The agent trains against past versions of itself, which are stored in a persistent league database relative to their ELO ratings.

## 📂 File Overview

### 1. `train_selfplay.py` (Main Training Script)
*   **Purpose:** The entry point for training.
*   **Function:** Loads best hyperparameters from Optuna, initializes a fresh PPO agent, and starts the training loop.
*   **League Integration:** Every 100,000 steps (configurable), it saves a snapshot of the current model (e.g., `gen_1.zip`) to the `league_models/` folder and registers it in the league database.

### 2. `league.py` (Database Manager)
*   **Purpose:** Manages the `league.json` database.
*   **Function:** 
    *   Stores ELO ratings and file paths for all agents.
    *   Handles safe concurrent updates using file locks (crucial for parallel training).
    *   Calculates ELO updates after matches.
    *   Samples opponents for new games.

### 3. `self_play_env.py` (Gym Environment Wrapper)
*   **Purpose:** Connects the training loop to the League.
*   **Function:** 
    *   Wraps the standard `CatanatronEnv`.
    *   In `reset()`, it asks the League for 3 opponents.
    *   Ensures the agent faces a dynamic mix of past selves and baselines.
    *   Reports match results back to the League to update ELO ratings.

### 4. `ppo_player.py` (Agent Wrapper)
*   **Purpose:** Allows saved PPO models to play valid moves in Catanatron games.
*   **Function:** Wraps a `MaskablePPO` model so it can be used as a standard `Player` object by the game engine. It handles observation conversion and action masking.

### 5. `league.json` (The Database)
*   **Purpose:** Persistent storage for the league.
*   **Content:** A JSON file tracking `elo`, `games` played, and `path` for every agent version.
    ```json
    {
        "random": {"type": "random", "elo": 1000},
        "gen_1": {"type": "ppo", "path": "league_models/gen_1.zip", "elo": 1050}
    }
    ```

---

## 🚀 How to Run

### Test
```
./venv/bin/python train_selfplay.py --steps 10000 --freq 2000 --envs 16 --name smoketest
```
### Real
```
./venv/bin/python train_selfplay.py --name real_run
```
