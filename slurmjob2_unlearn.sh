#!/bin/bash
#SBATCH --job-name=nova_job_unlearning # specify the job name for monitoring
#SBATCH --output=transformer-out/unlearning_JOB_%j.out # specify the output file
#SBATCH --error=transformer-err/unlearning_JOB_%j.err # specify the error file
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

python -c "import torch; print(torch.cuda.is_available())"

### Now you may start your operations below ###

# The actual command to run, using accelerate launch
# If you want make a testrun
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 trainer=GradAscent task_name=SAMPLE_UNLEARN
