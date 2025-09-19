#!/bin/bash

# Parameters
#SBATCH --error=/ictstr01/home/haicu/james.odgers/input_location_approximation/multirun/2025-09-19/10-16-46/.submitit/%j/%j_0_log.err
#SBATCH --job-name=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/ictstr01/home/haicu/james.odgers/input_location_approximation/multirun/2025-09-19/10-16-46/.submitit/%j/%j_0_log.out
#SBATCH --signal=USR2@120
#SBATCH --time=3
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /ictstr01/home/haicu/james.odgers/input_location_approximation/multirun/2025-09-19/10-16-46/.submitit/%j/%j_%t_log.out --error /ictstr01/home/haicu/james.odgers/input_location_approximation/multirun/2025-09-19/10-16-46/.submitit/%j/%j_%t_log.err /ictstr01/home/haicu/james.odgers/input_location_approximation/.venv/bin/python -u -m submitit.core._submit /ictstr01/home/haicu/james.odgers/input_location_approximation/multirun/2025-09-19/10-16-46/.submitit/%j
