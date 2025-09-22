#!/bin/bash
#SBATCH --job-name=TuneMaster # specify the job name for monitoring
#SBATCH --output=transformer-out/JOB_%j_finetuning_master.out # specify the output file
#SBATCH --error=transformer-err/JOB_%j_finetuning_master.err # specify the error file
#SBATCH --nodes=1 # As we have single node it should be always set as 1
#SBATCH --cpus-per-task=1 # Number of CPUs
#SBATCH --gres=gpu:1g.10gb:1  # Allocate 1 GPU resources with specified configurations
#SBATCH --mem=50G  # Specify the total amount of memory
#SBATCH --time=160:00:00
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
python hpsearch_setup.py
# This is a small helper script that queries the DB and prints the number of pending trials
python check_status.py > saves/status.txt

### First Worker is applicable
# Check the number of running worker jobs for this specific job array
RUNNING_JOBS=$(squeue -h -u $USER | wc -l)
# If we are below our parallel limit (e.g., 4) and there are more trials to run
  if [[ $RUNNING_JOBS -lt 5 && $(cat saves/status.txt) -gt 0 ]]; then
    # This worker will connect to the same database and pick the next available trial
    sbatch worker.sh
    RUNNING_JOBS=$(squeue -h -u $USER | wc -l)
    echo "Submitted a new worker job. Total Jobs: $RUNNING_JOBS"
  fi

# Wait a bit before checking again
sleep 60
# Update the trial status count
python check_status.py > saves/status.txt

# Check the number of running worker jobs for this specific job array
RUNNING_JOBS=$(squeue -h -u $USER | wc -l)
# Loop until all trials are complete (saves/status.txt will contain '0' or a smaller number than 4)
while [[ $(cat saves/status.txt) -gt 0 && $RUNNING_JOBS -gt 1 ]]; do

  # Check the number of running worker jobs for this specific job array
  RUNNING_JOBS=$(squeue -h -u $USER | wc -l)
  # If we are below our parallel limit (e.g., 4) and there are more trials to run
  if [[ $RUNNING_JOBS -lt 5 && $(cat saves/status.txt) -gt 0 ]]; then
    # This worker will connect to the same database and pick the next available trial
    sbatch worker.sh
    RUNNING_JOBS=$(squeue -h -u $USER | wc -l)
    echo "Submitted a new worker job. Total Jobs: $RUNNING_JOBS"
  fi
  if [[ $(cat saves/status.txt) -gt 0 && $RUNNING_JOBS -gt 4 ]]; then
    echo "Waiting on an open Job Slot. Total Jobs: $RUNNING_JOBS"
  fi

  # Wait a bit before checking again
  sleep 300
  
  # Update the trial status count
  python check_status.py > saves/status.txt

  # Check the number of running worker jobs for this specific job array
  RUNNING_JOBS=$(squeue -h -u $USER | wc -l)
  if [[ $(cat saves/status.txt) -eq 0 && $RUNNING_JOBS -gt 1 ]]; then
    echo "Submitted all possible Trials, Waiting on them to finish. Total Jobs: $RUNNING_JOBS"
  fi
done

echo "Grid search is complete. All trials have been run."