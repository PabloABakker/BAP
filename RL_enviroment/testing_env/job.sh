#!/bin/bash

#SBATCH --job-name=flapping_wing_rl_tuning
#SBATCH --output=ray_out_%j.log
#SBATCH --error=ray_err_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # Ray will use 8 CPUs
#SBATCH --mem-per-cpu=3G               # 8 * 4G = 32G total
#SBATCH --time=04:00:00
#SBATCH --partition=compute


# Load required modules
module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-gymnasium
module load py-ray
module load py-ray-tune
module load py-stable-baselines3
module load py-torch
module load py-tensorflow

# Ray memory monitor threshold
export RAY_MEMORY_MONITOR_ERROR_THRESHOLD=0.9

# Run Python script in tuning mode
srun python delftblue_rl.py --mode tune --env CustomDynamicsEnv-v2 --algo sac


