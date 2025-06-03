#!/bin/bash
#debugging job
#SBATCH --job-name=nova_job_unlearning # specify the job name for monitoring
#SBATCH --output=transformer-out/unlearning_JOB_%j.out # specify the output file
#SBATCH --error=transformer-err/unlearning_JOB_%j.err # specify the error file
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
# python src/eval.py \
#   experiment=eval/tofu/default.yaml \
#   forget_split=forget01 \
#   holdout_split=holdout01 \
#   model=Llama-2-7b-hf \
#   task_name=tofu_Llama-2-7b-hf_forget01_NPO \
#   model.model_args.pretrained_model_name_or_path=??? \
#   paths.output_dir=saves/unlearn/tofu_Llama-2-7b-hf_forget01_NPO/evals \
#   retain_logs_path=???

python src/train.py --config-name=train.yaml experiment=finetune/tofu/default \
  trainer.args.learning_rate=5e-5 task_name=Llama-3.2-3B-Instruct_finetune_example \
  paths.output_dir=saves/unlearn/Llama-3.2-3B-Instruct-hf_forget01_NPO/evals \
  retain_logs_path=saves/eval/tofu_Llama-3.2-3B-Instruct_retain99/TOFU_EVAL.json