# Training Guide: From Testing to Long-Haul Deployment

This guide explains how to robustly test your self-play training setup and then deploy it for a multi-day training session.

## 1. The "Smoke Test" (Verification)
Before committing to a long run, verify that everything works correctly (league updates, model saving, no crashes).

Run this command to train for a very short duration:

```bash
./venv/bin/python train_selfplay.py --steps 50000 --freq 10000 --envs 4 --name smoke_test
```

**What to check:**
*   **No Crashes**: Ensure it runs to completion or for at least a few minutes.
*   **League Updates**: Check `league_models/`. You should see `smoke_test_gen_1.zip`, `smoke_test_gen_2.zip`.
*   **Model Saving**: Check the `league.json`. It should have new entries.

---

## 2. The Long Haul (Background Execution)
Once the test passes, start the real training. We use `nohup` so the process keeps running even if you close the terminal or disconnect your SSH session.

Run this command:

```bash
nohup ./venv/bin/python train_selfplay.py --steps 100000000 --freq 100000 --envs 8 --name main_run > training.log 2>&1 &
```

**Breakdown of the command:**
*   `nohup ... &`: Runs the command in the background and ignores "hangup" signals (closing terminal).
*   `--steps 100000000`: Sets training to 100 million steps (likely several days).
*   `--freq 100000`: Adds a new model to the league every 100k steps.
*   `--envs 8`: Uses 8 parallel environments to balance speed and VRAM usage.
*   `--name main_run`: Gives checkpoints a prefix (e.g., `main_run_gen_1.zip`) so you can distinguish them from other runs.
*   `> training.log 2>&1`: Redirects all output (standard output and errors) to `training.log`.

---

## 3. Switching & Resuming Runs (IMPORTANT)

**Question:** If I train Model A, stop, train Model B, then resume Model A, will it break?
**Answer:** No, provided you use different `--name` arguments.

### Scenario: Train Model A
```bash
./venv/bin/python train_selfplay.py --name model_A
```
*Generates: `model_A_gen_1.zip`, `model_A_gen_2.zip`...*

### Scenario: Train Model B (Start Fresh)
```bash
./venv/bin/python train_selfplay.py --name model_B
```
*Generates: `model_B_gen_1.zip`, `model_B_gen_2.zip`...*
*Note: Model B will play against A's snapshots in the league!*

### Scenario: Resume Model A
To resume Model A from where it left off:
1.  Find the latest checkpoint (e.g., `league_models/model_A_gen_50.zip`).
2.  Run with `--load` AND the same `--name`:

```bash
./venv/bin/python train_selfplay.py --load league_models/model_A_gen_50.zip --name model_A
```
*The script automatically details the next generation number (e.g., 51) based on existing files, so it won't overwrite anything.*

---

## 4. Monitoring & Management

### Check Status (Logs)
To see the latest output from your running process:

```bash
tail -f training.log
```
*(Press `Ctrl+C` to stop watching the log. The training will continue running.)*

### Monitor Performance (Graphs)
Visualize win rates, rewards, and losses in real-time:

```bash
./venv/bin/tensorboard --logdir runs/selfplay
```
Then open `http://localhost:6006` in your browser.

### Stop Training
If you need to stop the background process:

1.  **Find the Process ID (PID):**
    ```bash
    ps aux | grep train_selfplay.py
    ```
    You will see a line like `x 12345 ... python train_selfplay.py`. The number `12345` is the PID.

2.  **Kill the Process:**
    ```bash
    kill 12345
    ```
