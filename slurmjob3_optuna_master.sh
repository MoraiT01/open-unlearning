#!/bin/bash
#SBATCH --job-name=finetune_master # specify the job name for monitoring
#SBATCH --output=transformer-out/finetuning_master_JOB_%j.out # specify the output file
#SBATCH --error=transformer-err/finetuning_master_JOB_%j.err # specify the error file
#SBATCH --nodes=1 # As we have single node it should be always set as 1
#SBATCH --cpus-per-task=1 # Number of CPUs
#SBATCH --gres=gpu:1g.10gb:1  # Allocate 1 GPU resources with specified configurations
#SBATCH --mem=50G  # Specify the total amount of memory
#SBATCH --time=168:00:00
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

python create_db.py
# This is a small helper script that queries the DB and prints the number of pending trials
python check_status.py > status.txt

# Loop until all trials are complete (status.txt will contain '0' or a smaller number than 4)
while [[ $(cat status.txt) -gt 0 ]]; do

  # Check the number of running worker jobs for this specific job array
  RUNNING_JOBS=$(squeue -h -u $USER -n nova_worker | wc -l)

  # If we are below our parallel limit (e.g., 4) and there are more trials to run
  if [[ $RUNNING_JOBS -lt 4 ]]; then
    echo "Submitting a new worker job. Running jobs: $RUNNING_JOBS"
    # This worker will connect to the same database and pick the next available trial
    sbatch worker.sh
  fi

  # Wait a bit before checking again
  sleep 60
  
  # Update the trial status count
  python check_status.py > status.txt

done

echo "Grid search is complete. All trials have been run."