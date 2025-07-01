import sys
import pickle
import optuna
import os
import subprocess
import json
import torch
import math
import time

import logging
import os # Import os for path manipulation

# Constants for the experiment
BASE_MODEL = "Llama-3.2-1B-Instruct"
# Out of the following models: [Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct, Llama-3.2-1B-Instruct]
FINETUNED_MODEL_OUTPUT_PATH = f"open-unlearning/tofu_{BASE_MODEL}_full" # Path to store the initially finetuned model
FORGET_SPLIT = "forget10"
RETAIN_SPLIT = "retain90"
HOLDOUT_SPLIT = "holdout10" # Used in eval pipeline
# Path to reference retain logs, assuming they are downloaded via setup_data.py
RETAIN_LOGS_PATH = f"saves/eval/tofu_{BASE_MODEL}_{RETAIN_SPLIT}/TOFU_EVAL.json"
MAXIMIZE_FORGETTING = False
KEEP_MODEL_TENSORS = False

# Assuming the setup_data.py was executed before
initial_finetune_eval_output_dir = f"saves/eval/tofu_{BASE_MODEL}_full/evals_{FORGET_SPLIT}"
INITIAL_FINETUNE_SUMMARY_FILE_PATH = os.path.join(initial_finetune_eval_output_dir, "TOFU_SUMMARY.json")


# Configure a logger for nova_optuna.py
logger = logging.getLogger("HPTUNER")
logger.setLevel(logging.INFO) # Set the logging level (e.g., INFO, DEBUG, WARNING)

# Create a formatter
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')

# Create a file handler
log_file_path = f"logs/opti_{BASE_MODEL}_{FORGET_SPLIT}.log" # You can make this dynamic if needed
if MAXIMIZE_FORGETTING:
    log_file_path = f"logs/opti_{BASE_MODEL}_{FORGET_SPLIT}_maxf.log" 
file_handler = logging.FileHandler(log_file_path, mode="a")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Optionally, add a stream handler to output to console as well
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def optuna_setup():
    # Define study name and storage for Optuna

    study_name = f"OptiNOVA_{BASE_MODEL}_{FORGET_SPLIT}"
    if MAXIMIZE_FORGETTING:
        study_name = f"OptiNOVA_{BASE_MODEL}_{FORGET_SPLIT}_maxforgetting"
    storage_name = "sqlite:///{}.db".format("HP_Opti_NOVA")

    # Create or load the Optuna study. 'minimize' direction is set as our objective is to minimize a combined metric.
    sampler_name = f"sampler_nova_{BASE_MODEL}_{FORGET_SPLIT}_maxf.pkl" if MAXIMIZE_FORGETTING == True else f"sampler_nova_{BASE_MODEL}_{FORGET_SPLIT}.pkl"
    if os.path.exists(sampler_name):
        restored_sampler = pickle.load(open(sampler_name, "rb"))
        study_nova = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize", load_if_exists=True, sampler=restored_sampler,)
    else:
        study_nova = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize", load_if_exists=True,)

    # Set HF_HOME environment variable for consistent caching across runs
    os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

    # --- Phase 1: Get Metrics of BASE_MODEL ---
    logger.info(f"--- Grab the metrics for {BASE_MODEL} ---")

    try:
        # Load baseline metrics
        if not os.path.exists(INITIAL_FINETUNE_SUMMARY_FILE_PATH):
            raise FileNotFoundError(f"Initial finetune summary file not found: {INITIAL_FINETUNE_SUMMARY_FILE_PATH}")
        with open(INITIAL_FINETUNE_SUMMARY_FILE_PATH, 'r') as f:
            initial_metrics = json.load(f)
        baseline_model_utility = initial_metrics.get("model_utility")
        logger.info(f"Initial Finetuned Model Utility: {baseline_model_utility}")

    except Exception as e:
        logger.error(f"Error during initial finetuning or evaluation: {e}")
        sys.exit(1) # Exit if initial setup fails
    
    return study_nova, sampler_name

def scale_to_0_1(original_value, x1, y1):
    """
    Scales a value from the interval [x1, y1] to the interval [0, 1].

    Args:
        original_value: The value to be scaled.
        x1: The lower bound of the original interval.
        y1: The upper bound of the original interval.

    Returns:
        The scaled value between 0 and 1.

    Raises:
        ValueError: If y1 is not greater than x1 (to avoid division by zero).
    """
    if y1 <= x1:
        raise ValueError("y1 must be greater than x1 for scaling.")
    
    scaled_value = (original_value - x1) / (y1 - x1)
    return scaled_value

# --- Phase 2: Optuna Optimization Loop ---
def objective(trial):

    # Hyperparameters to be optimized by Optuna for the NOVA algorithm
    opt_noise_epochs = trial.suggest_int("noise_epochs", 1, 100)
    opt_noise_lr = trial.suggest_float("noise_lr", 0.0000001, 1.0, log=True) # Log scale for learning rate
    opt_regularization_term = trial.suggest_float("regularization_term", 0.0000001, 1.0, log=True) # Log scale for regularization
    opt_impair_gamma = trial.suggest_float("impair_gamma", 0.000001, 10.0, log=True) # Log scale for gamma
    opt_repair_alpha = trial.suggest_float("repair_alpha", 0.000001, 10.0, log=True) # Log scale for alpha

    # Generate a unique task name for the current trial to store results separately
    trial_task_name = f"nova_trial_{trial.number}" # _ne{opt_noise_epochs}_nlr{opt_noise_lr:.5f}_reg{opt_regularization_term:.5f}_g{opt_impair_gamma:.5f}_a{opt_repair_alpha:.5f}"
    unlearn_output_dir = f"saves/unlearn/default/{BASE_MODEL}/{FORGET_SPLIT}/{trial_task_name}"
    if MAXIMIZE_FORGETTING:
        unlearn_output_dir = f"saves/unlearn/maxforgetting/{BASE_MODEL}/{FORGET_SPLIT}/{trial_task_name}"
    eval_output_dir = f"{unlearn_output_dir}/evals"
    summary_file_path = os.path.join(eval_output_dir, "TOFU_SUMMARY.json")

    # Create necessary output directories for the trial
    os.makedirs(unlearn_output_dir, exist_ok=True)
    os.makedirs(eval_output_dir, exist_ok=True)

    # Construct the training command for unlearning with NOVA (without accelerate)
    train_command = [
        "python", "src/train.py",
        "--config-name=unlearn.yaml",
        "experiment=unlearn/tofu/default", # Use the default TOFU unlearn experiment config
        f"trainer=NOVA", # Specify NOVA as the trainer
        f"task_name={trial_task_name}",
        f"model={BASE_MODEL}",
        f"forget_split={FORGET_SPLIT}",
        f"retain_split={RETAIN_SPLIT}",
        f"retain_logs_path={RETAIN_LOGS_PATH}",
        # Pass the Optuna suggested hyperparameters to the trainer's method_args
        f"trainer.method_args.noise_epochs={opt_noise_epochs}",
        f"trainer.method_args.noise_lr={opt_noise_lr}",
        f"trainer.method_args.regularization_term={opt_regularization_term}",
        f"trainer.method_args.impair_gamma={opt_impair_gamma}",
        f"trainer.method_args.repair_alpha={opt_repair_alpha}",
        f"paths.output_dir={unlearn_output_dir}", # Set dynamic output path for the model checkpoint
    ]

    # Construct the evaluation command
    eval_command = [
        "python", "src/eval.py",
        "experiment=eval/tofu/default.yaml", # Use the default TOFU evaluation config
        f"forget_split={FORGET_SPLIT}",
        f"holdout_split={HOLDOUT_SPLIT}",
        f"model={BASE_MODEL}",
        f"task_name={trial_task_name}",
        f"model.model_args.pretrained_model_name_or_path={unlearn_output_dir}", # Path to the unlearned model from training
        f"paths.output_dir={eval_output_dir}", # Set dynamic output path for evaluation logs
        f"retain_logs_path={RETAIN_LOGS_PATH}" # Path to reference retain logs for metrics like forget_quality
    ]

    logger.info(f"### Starting Training Process for Trial {trial.number} ###")
    logger.info(f"{' '.join(train_command)}")
    try:
        # Execute the training command
        start_time = time.time()
        train_process = subprocess.run(train_command, check=False, capture_output=True, text=True)
        if train_process.returncode != 0:
            logger.error(f"Train command failed for trial {trial.number}. STDOUT:\n{train_process.stdout}\nSTDERR:\n{train_process.stderr}")
            raise RuntimeError("Training process failed")
        training_datetime = time.time()

        logger.info(f"### Starting Evaluation Process for Trial {trial.number} ###")
        logger.info(f"{' '.join(eval_command)}")
        # Execute the evaluation command
        eval_process = subprocess.run(eval_command, check=False, capture_output=True, text=True)
        if eval_process.returncode != 0:
            logger.error(f"Eval command failed for trial {trial.number}. STDOUT:\n{eval_process.stdout}\nSTDERR:\n{eval_process.stderr}")
            raise RuntimeError("Evaluation process failed")

        # Read the evaluation summary results
        if not os.path.exists(summary_file_path):
            raise FileNotFoundError(f"Summary file not found: {summary_file_path}")

        with open(summary_file_path, 'r') as f:
            metrics = json.load(f)

        # Extract the desired metrics from the summary
        # forget_Q_A_Prob = metrics.get("forget_Q_A_Prob")
        forget_quality = metrics.get("forget_quality")
        model_utility = metrics.get("model_utility")

        if forget_quality is None or model_utility is None:
            raise ValueError("Required metrics (forget_quality or model_utility) not found in summary file.")
        
        if not os.path.exists(INITIAL_FINETUNE_SUMMARY_FILE_PATH):
            raise FileNotFoundError(f"Initial finetune summary file not found: {INITIAL_FINETUNE_SUMMARY_FILE_PATH}")  
        with open(INITIAL_FINETUNE_SUMMARY_FILE_PATH, 'r') as f:
            initial_metrics = json.load(f)
        baseline_forget_quality = initial_metrics.get("forget_quality")
        baseline_model_utility = initial_metrics.get("model_utility")
        
        if baseline_forget_quality is None or baseline_model_utility is None:
            raise ValueError("Required metrics (baseline_forget_quality or baseline_model_utility) not found in summary file.")

        delta_model_utility = abs(baseline_model_utility - model_utility)

        # TOFU paper also presented the f_q on log scale, for better comparability
        log_baseline_forget_quality = math.log10(baseline_forget_quality)
        log_forget_quality          = math.log10(forget_quality)

        scaled_forget_quality = scale_to_0_1(log_forget_quality, log_baseline_forget_quality, 0) # log10(1) = 0, which is the best thing that can happen
        scaled_delta_model_utility  = scale_to_0_1(delta_model_utility, 0, max(baseline_model_utility, 1 - baseline_model_utility))
        # scaled_forget_quality       = forget_quality
        # scaled_delta_model_utility  = delta_model_utility

        # Define the objective value to maximize: (forget_quality - delta_model_utility)
        # higher forget_quality is better (more unlearning), lower difference in model_utility is better (more original utility retained).
        # So, maximizing this difference encourages both.
        # Calculate the objective value with the new function
        # For Testing on "MaxForgetting" we remove the model utility term
        objective_value = scaled_forget_quality - scaled_delta_model_utility
        if MAXIMIZE_FORGETTING:
            objective_value = scaled_forget_quality 

        logger.info(f"Trial {trial.number} completed. forget_quality: {scaled_forget_quality}, delta_model_utility: {scaled_delta_model_utility}, Objective: {objective_value}")
        
        # --- Report non-looping, trial-specific information ---
        # Store training time
        training_time = training_datetime - start_time
        trial.set_user_attr("Training Time (secs)", training_time)
        # Store Forget Quality
        trial.set_user_attr("Forget Quality (log10-scale)", log_forget_quality)
        # Store Model Utility
        trial.set_user_attr("Model Utility", model_utility)

        # Delete the model if needed
        if not KEEP_MODEL_TENSORS:
            # Path to the model.tensors file
            model_tensors_file_path = os.path.join(unlearn_output_dir, "model.safetensors")

            # --- Add these lines for cleanup ---
            if os.path.exists(model_tensors_file_path):
                try:
                    os.remove(model_tensors_file_path)
                    logger.info(f"Successfully deleted {model_tensors_file_path} for trial {trial.number}.")
                except OSError as e:
                    logger.warning(f"Error deleting {model_tensors_file_path} for trial {trial.number}: {e}")
            # --- End of cleanup lines ---

    except Exception as e:
        logger.warning(f"Trial {trial.number} failed due to an error: {e}; Returning '-inf' loss")
        # Return a very large value to penalize trials that fail or encounter errors
        return float('-inf')

    return objective_value

def main(optuna_tuning: bool = True):
    # Call optuna_setup to initialize the study and get baseline metrics
    study_nova, sampler_name = optuna_setup()

    # Start Trials
    if optuna_tuning:
        study_nova.optimize(objective, n_trials=50)

    # Save the Optuna sampler state for resuming the study later if needed
    with open(sampler_name, "wb") as fout:
        pickle.dump(study_nova.sampler, fout)

if __name__ == "__main__":
    logger.info("Starting the Hyperparameter Tuning process for Optuna.")
    main()
