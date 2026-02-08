#!/bin/bash
#SBATCH --job-name=catan_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate your python environment
# source /path/to/your/venv/bin/activate
# OR load modules
# module load python/3.10 cuda/12.0

echo "Starting training on $(hostname)"
echo "Date: $(date)"

# Run the training script
# Arguments: total_timesteps, n_envs
# We set n_envs to 16 to match cpus-per-task
srun python3 gym-agent.py 10000000 16

echo "Training finished at $(date)"
