#!/bin/bash
#debugging job
#SBATCH --job-name=nova_job_hpfinetuning # specify the job name for monitoring
#SBATCH --output=transformer-out/hpfinetuning_JOB_%j.out # specify the output file
#SBATCH --error=transformer-err/hpfinetuning_JOB_%j.err # specify the error file
#SBATCH --nodes=1 # As we have single node it should be always set as 1
#SBATCH --cpus-per-task=4 # Number of CPUs
#SBATCH --gres=gpu:1g.10gb:1  # Allocate 1 GPU resources with specified configurations
#SBATCH --mem=50G  # Specify the total amount of memory
#SBATCH --time=72:00:00  # Set the time limit to 72 hours
#SBATCH --partition=debugging 
#SBATCH --qos=debugging
#SBATCH --account=debugging


# Run the Python script
srun hostname

# If you still need to setup the environment:
# bash slurmjob0_install.sh

# Initialize Conda for the current shell session
# Replace '~/miniconda3' with the actual path to your Miniconda/Anaconda installation if different
source /fast_storage/kastler/miniconda3/etc/profile.d/conda.sh

# Now activate your environment
conda activate /fast_storage/kastler/miniconda3/envs/unlearning

# Verify activation
conda info --envs

### Now you may start your operations below ###