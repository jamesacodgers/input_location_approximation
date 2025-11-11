#!/bin/bash
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --nice=1000
#SBATCH --job-name=mfvi_experiment
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Activate your virtual environment if needed (optional, since you're using .venv/bin/python directly)
# source .venv/bin/activate

# Run your command
.venv/bin/python main.py -m \
  posterior=mfvi \
  optimization.epochs=80_000 \
  posterior.temperature=1e-3,1e-2,1e-1,1 \
  seed=1 \
  dataset.n_train=100,1000 \
  dataset.frequency=0.25,0.5,1,2,4,8 \



