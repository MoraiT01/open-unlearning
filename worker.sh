#!/bin/bash
#SBATCH --job-name=nova_job_hpfinetuning # specify the job name for monitoring
#SBATCH --output=transformer-out/hpfinetuning_JOB_%j.out # specify the output file
#SBATCH --error=transformer-err/hpfinetuning_JOB_%j.err # specify the error file
#SBATCH --nodes=1 # As we have single node it should be always set as 1
#SBATCH --cpus-per-task=10 # Number of CPUs
#SBATCH --gres=gpu:7g.79gb:1  # Allocate 1 GPU resources with specified configurations
#SBATCH --mem=400G  # Specify the total amount of memory
#SBATCH --time=96:00:00
#SBATCH --partition=ultimate
#SBATCH --qos=ultimate
#SBATCH --account=ultimate


# Run the Python script
srun hostname

# If you still need to setup the environment:
# bash slurmjob0_install.sh

# Initialize Conda for the current shell session
# Replace '~/miniconda3' with the actual path to your Miniconda/Anaconda installation if different
source /fast_storage/kastler/miniconda3/etc/profile.d/conda.sh

# Now activate your environment
conda activate /fast_storage/kastler/miniconda3/envs/unlearning


### Now you may start your operations below ###

# Run your single-trial Python script
# The script will connect to the Optuna DB, find the next available trial,
# and run it. The DB handles the coordination.
python nova_optuna.py