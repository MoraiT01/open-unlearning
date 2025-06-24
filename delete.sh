#!/bin/bash
#debugging job
#SBATCH --job-name=mu_delete_models_tensors # specify the job name for monitoring
#SBATCH --output=transformer-out/delete_models_tensors_JOB_%j.out # specify the output file
#SBATCH --error=transformer-err/delete_models_tensors_JOB_%j.err # specify the error file
#SBATCH --nodes=1 # As we have single node it should be always set as 1
#SBATCH --cpus-per-task=4 # Number of CPUs
#SBATCH --gres=gpu:1g.10gb:1  # Allocate 1 GPU resources with specified configurations
#SBATCH --mem=50G  # Specify the total amount of memory
#SBATCH --time=72:00:00  # Set the time limit to 72 hours
#SBATCH --partition=debugging 
#SBATCH --qos=debugging
#SBATCH --account=debugging


# This script finds and deletes all files named "model.tensors"
# within a specified directory and all its subdirectories.

# --- Configuration ---
# Set the target directory where you want to start the search.
# IMPORTANT: Replace "/path/to/your/main/directory" with the actual path.
# Example: TARGET_DIR="/home/user/my_unlearning_saves"
TARGET_DIR="saves/unlearn/DELETE"

# --- Script Logic ---

echo "Searching for 'model.tensors' files in: $TARGET_DIR"
echo "---"

# Check if the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory '$TARGET_DIR' not found."
    exit 1
fi

# Find and delete files named "model.tensors"
# -name "model.tensors": Searches for files with this specific name.
# -type f: Ensures only regular files are considered (not directories).
# -delete: Deletes the found files.
# Alternatively, to see what would be deleted without actually deleting (dry run):
# find "$TARGET_DIR" -name "model.tensors" -type f -print
# To delete with confirmation:
# find "$TARGET_DIR" -name "model.tensors" -type f -ok rm {} \;

find "$TARGET_DIR" -name "model.safetensors" -type f -print -delete

# The -print option will show the files being deleted.
# If you don't want to see the output for each deleted file, remove `-print`.

echo "---"
echo "Deletion process complete."