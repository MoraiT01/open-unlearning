#!/bin/bash
#debugging job
#SBATCH --job-name=nova_job_install_conda_env # specify the job name for monitoring
#SBATCH --output=transformer-out/install_conda_env_JOB_%j.out # specify the output file
#SBATCH --error=transformer-err/install_conda_env_JOB_%j.err # specify the error file
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

# print MIG devices ids for debugging
echo $CUDA_VISIBLE_DEVICES

# Add this line early in your Slurm script
source $(conda info --base)/etc/profile.d/conda.sh

# Needed until the environment runs smoothly
# Optional: Remove the old environment cleanly first to avoid prompts
conda remove -n unlearning --all -y

# Create with Python 3.11 and auto-confirm with -y
conda create -n unlearning python=3.11 -y 
conda activate unlearning

# chech if packaging was successfully installed
python --version
pip install -r requirements.txt
pip install .[lm_eval]
pip install --no-build-isolation flash-attn==2.6.3