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
  posterior=sbvi \
  posterior.num_samples=1,4,16 \
  optimization.epochs=30_000 \
  posterior.temperature=1,1e-3,1e-6 \
  posterior.num_squash_vectors=1,4,16 \
  seed=0 \
  optimization.learning_rate=3e-4,1e-4 \
  dataset.n_train=4,64,512 \
  
  



