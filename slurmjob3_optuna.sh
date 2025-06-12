#!/bin/bash
#SBATCH --job-name=nova_job_hpfinetuning # specify the job name for monitoring
#SBATCH --output=transformer-out/hpfinetuning_JOB_%j.out # specify the output file
#SBATCH --error=transformer-err/hpfinetuning_JOB_%j.err # specify the error file
#SBATCH --nodes=1 # As we have single node it should be always set as 1
#SBATCH --cpus-per-task=4 # Number of CPUs
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.39gb:1  # Allocate 1 GPU resources with specified configurations
#SBATCH --mem=200G  # Specify the total amount of memory
#SBATCH --time=50:00:00  # Set the time limit to 72 hours
#SBATCH --partition=advance
#SBATCH --qos=advance
#SBATCH --account=advance


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

# Set MASTER_PORT if needed for internal processes that might use it, though direct python calls are used now.
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Ensure HF_HOME is set for consistent caching
export HF_HOME=$HOME/.cache/huggingface

# Run the Optuna optimization script.
python nova_optuna.py

echo "Slurm job finished at: $(date)"