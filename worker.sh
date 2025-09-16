#!/bin/bash
#SBATCH --job-name=TuneWorker # specify the job name for monitoring
#SBATCH --output=transformer-out/workers/JOB_%j_finetuning_worker.out # specify the output file
#SBATCH --error=transformer-err/workers/JOB_%j_finetuning_worker.err # specify the error file
#SBATCH --nodes=1 # As we have single node it should be always set as 1
#SBATCH --cpus-per-task=10 # Number of CPUs
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.39gb:1  # Allocate 1 GPU resources with specified configurations
#SBATCH --mem=300G  # Specify the total amount of memory
#SBATCH --time=2:00:00
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

export OMP_THREAD_AFFINITY=FALSE
### Now you may start your operations below ###

# Run your single-trial Python script
# The script will connect to the Optuna DB, find the next available trial,
# and run it. The DB handles the coordination.
python nova_optuna.py