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

set -e  # Exit immediately if a command exits with a non-zero status
# Run the Python script
srun hostname

# print MIG devices ids for debugging
echo $CUDA_VISIBLE_DEVICES

source $(conda info --base)/etc/profile.d/conda.sh

# Needed until the environment runs smoothly
# Optional: Remove the old environment cleanly first to avoid prompts
conda remove -n unlearning --all -y

# Create with Python 3.11 and auto-confirm with -y
conda create -n unlearning python=3.11 -y 
conda activate unlearning

mkdir -p $CONDA_PREFIX/pip-config
cat > $CONDA_PREFIX/pip-config/pip.conf << 'EOF'
[global]
no-cache-dir = true
index-url = https://pypi.org/simple
extra-index-url =
trusted-host =
EOF

conda env config vars set PIP_CONFIG_FILE=$CONDA_PREFIX/pip-config/pip.conf
conda deactivate && conda activate unlearning

# chech if packaging was successfully installed
python --version
# Install the required dependencies
# pip install -r requirements.txt
pip install ".[lm_eval]"

# pip install --no-build-isolation flash-attn==2.6.3
pip install "https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/wheelhouse-flash_attn-2.8.3/linux_x86_64/torch2.4/cu12/abiFALSE/cp311/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# Data setup
# python setup_data.py --eval 

pip install optuna==4.3.0
pip install optuna-dashboard==0.18.0