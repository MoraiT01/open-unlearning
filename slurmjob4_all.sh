#!/bin/bash
#SBATCH --job-name=run_all_unlearning_combinations # specify the job name for monitoring
#SBATCH --output=transformer-out/run_all_unlearning_combinations_JOB_%j.out # specify the output file
#SBATCH --error=transformer-err/run_all_unlearning_combinations_JOB_%j.err # specify the error file
#SBATCH --nodes=1 # As we have single node it should be always set as 1
#SBATCH --cpus-per-task=20 # Number of CPUs
#SBATCH --gres=gpu:7g.79gb:1  # Allocate 1 GPU resources with specified configurations
#SBATCH --mem=300G  # Specify the total amount of memory
#SBATCH --time=72:00:00  # Set the time limit to 72 hours
#SBATCH --partition=ultimate
#SBATCH --qos=ultimate
#SBATCH --account=ultimate

# Run the Python script
srun hostname

# If you still need to setup the environment:
# slurmjob0_install.sh

# Initialize Conda for the current shell session
# Replace '~/miniconda3' with the actual path to your Miniconda/Anaconda installation if different
source /fast_storage/kastler/miniconda3/etc/profile.d/conda.sh

# Now activate your environment
conda activate /fast_storage/kastler/miniconda3/envs/unlearning



# Verify activation
conda info --envs

### Now you may start your operations below ###

echo "Starting bulk unlearning runs..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"

# --- Define lists for iteration ---
# declare -a algorithms=("GradAscent" "GradDiff" "NPO" "DPO" "SimNPO" "RMU" "UNDIAL" "NOVA")
declare -a algorithms=("NOVA")
declare -a models=("Llama-3.1-8B-Instruct" "Llama-3.2-3B-Instruct" "Llama-3.2-1B-Instruct")
declare -a forget_splits=("forget10" "forget05" "forget01")

# Constants (as per your Optuna script)
RETAIN_SPLIT="retain90"
HOLDOUT_SPLIT="holdout10"

# --- Loop through combinations ---
for ALGO in "${algorithms[@]}"; do
    for MODEL in "${models[@]}"; do
        for FORGET_SPLIT_NAME in "${forget_splits[@]}"; do
            echo "--- Processing: Algo=${ALGO}, Model=${MODEL}, ForgetSplit=${FORGET_SPLIT_NAME} ---"

            # Extract the numerical part from FORGET_SPLIT_NAME (e.g., "10" from "forget10")
            # This assumes the format is always "forgetXX"
            FORGET_PERCENTAGE=$(echo "$FORGET_SPLIT_NAME" | sed 's/forget//')

            # Calculate RETAIN_PERCENTAGE
            RETAIN_PERCENTAGE=$((100 - FORGET_PERCENTAGE))

            # Construct the dynamic RETAIN_SPLIT and HOLDOUT_SPLIT
            DYNAMIC_RETAIN_SPLIT="retain${RETAIN_PERCENTAGE}"
            DYNAMIC_HOLDOUT_SPLIT="holdout${FORGET_PERCENTAGE}"

            # Dynamically set paths based on current iteration
            CURRENT_RETAIN_LOGS_PATH="saves/eval/tofu_${MODEL}_${DYNAMIC_RETAIN_SPLIT}/TOFU_EVAL.json"
            
            # Shorten model name for directory path if it's too long
            MODEL_NAME_SHORT=$(echo "$MODEL" | sed 's/Llama-3\.1-//g; s/Llama-3\.2-//g; s/-Instruct//g')

            # Directory for this specific run's outputs
            # Using current timestamp to ensure unique directory names if runs are repeated
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            RUN_TASK_NAME="${ALGO}_${MODEL_NAME_SHORT}_${FORGET_SPLIT_NAME}_${TIMESTAMP}"
            
            # Base directory for outputs of this specific unlearning run
            UNLEARN_OUTPUT_BASE="saves/unlearn/bulk_run_post/${ALGO}/${MODEL}/${FORGET_SPLIT_NAME}/${RUN_TASK_NAME}"
            EVAL_OUTPUT_DIR="${UNLEARN_OUTPUT_BASE}/evals"
            
            mkdir -p "$UNLEARN_OUTPUT_BASE"
            mkdir -p "$EVAL_OUTPUT_DIR"

            echo "Output directory: $UNLEARN_OUTPUT_BASE"
            echo "Eval output directory: $EVAL_OUTPUT_DIR"

            # --- Construct Training Command ---
            TRAIN_COMMAND=(
                "python" "src/train.py"
                "--config-name=unlearn.yaml"
                "experiment=unlearn/tofu/default"
                "trainer=${ALGO}"
                "task_name=${RUN_TASK_NAME}"
                "model=${MODEL}"
                "forget_split=${FORGET_SPLIT_NAME}"
                "retain_split=${DYNAMIC_RETAIN_SPLIT}"
                "retain_logs_path=${CURRENT_RETAIN_LOGS_PATH}"
                "paths.output_dir=${UNLEARN_OUTPUT_BASE}"
            )
            echo "Running training command:"
            echo "${TRAIN_COMMAND[@]}"
            "${TRAIN_COMMAND[@]}"
            TRAIN_EXIT_CODE=$?

            if [ $TRAIN_EXIT_CODE -ne 0 ]; then
                echo "ERROR: Training failed for ${ALGO}-${MODEL}-${FORGET_SPLIT_NAME}. Skipping evaluation."
                continue # Skip to next combination
            fi

            # --- Construct Evaluation Command ---
            EVAL_COMMAND=(
                "python" "src/eval.py"
                "experiment=eval/tofu/default.yaml"
                "forget_split=${FORGET_SPLIT_NAME}"
                "holdout_split=${DYNAMIC_HOLDOUT_SPLIT}"
                "model=${MODEL}"
                "task_name=${RUN_TASK_NAME}"
                "model.model_args.pretrained_model_name_or_path=${UNLEARN_OUTPUT_BASE}"
                "paths.output_dir=${EVAL_OUTPUT_DIR}"
                "retain_logs_path=${CURRENT_RETAIN_LOGS_PATH}"
            )
            echo "Running evaluation command:"
            echo "${EVAL_COMMAND[@]}"
            "${EVAL_COMMAND[@]}"
            EVAL_EXIT_CODE=$?

            if [ $EVAL_EXIT_CODE -ne 0 ]; then
                echo "ERROR: Evaluation failed for ${ALGO}-${MODEL}-${FORGET_SPLIT_NAME}."
            else
                echo "SUCCESS: ${ALGO}-${MODEL}-${FORGET_SPLIT_NAME} completed successfully."
                # Optional: Clean up model.safetensors or other large files
                # if you don't need them after evaluation
                # MODEL_TENSORS_FILE="$UNLEARN_OUTPUT_BASE/model.safetensors"
                # if [ -f "$MODEL_TENSORS_FILE" ]; then
                #     echo "Deleting: $MODEL_TENSORS_FILE"
                #     rm "$MODEL_TENSORS_FILE"
                # fi
                # For the next Run, I'd like to keep the models
            fi
            echo "" # Add a newline for readability between runs
            #!/bin/bash

            # Configuration
            CHROMADB_ID="default"
            BASE_DIR="saves/chromadb"
            TEMP_DIR="${BASE_DIR}/${CHROMADB_ID}"

            # Check if an ID was provided
            if [ -z "$CHROMADB_ID" ]; then
                echo "❌ Error: No chromadb_id provided."
                exit 1
            fi

            # Logic to delete the directory
            if [ -d "$TEMP_DIR" ]; then
                if rm -rf "$TEMP_DIR"; then
                    echo "✅ Successfully cleaned up ephemeral session: ${TEMP_DIR}"
                else
                    echo "❌ Failed to clean up temporary directory: ${TEMP_DIR}"
                    exit 1
                fi
            else
                echo "ℹ️ Directory does not exist: ${TEMP_DIR}"
            fi
        done
    done
done

echo "All bulk unlearning runs finished."
