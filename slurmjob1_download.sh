#!/bin/bash
#debugging job
#SBATCH --job-name=debug_movie_review_job # specify the job name for monitoring
#SBATCH --output=transformer-out/moviereview_JOB_%j.out # specify the output file
#SBATCH --error=transformer-err/moviereview_JOB_%j.err # specify the error file
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

bash slurmjob0_install.sh

# Logging
conda env list

### Now you may start your operations below ###
# Data setup
python setup_data.py --eval # saves/eval now contains evaluation results of the uploaded models
# This downloads log files with evaluation results (including retain model logs)
# into `saves/eval`, used for evaluating unlearning across supported benchmarks.
# Additional datasets (e.g., WMDP) are supported — run below for options:
python setup_data.py --help
