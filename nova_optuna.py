import sys
import pickle
import optuna
import os
import subprocess
import json
import torch
import math 

import logging
# Configure basic logging to stdout with INFO level
logger = logging.getLogger(__name__)

# Constants for the experiment
BASE_MODEL = "Llama-3.2-1B-Instruct"
# Out of the following models:
FINETUNED_MODEL_OUTPUT_PATH = f"open-unlearning/tofu_{BASE_MODEL}_full" # Path to store the initially finetuned model
FORGET_SPLIT = "forget10"
RETAIN_SPLIT = "retain90"
HOLDOUT_SPLIT = "holdout10" # Used in eval pipeline
# Path to reference retain logs, assuming they are downloaded via setup_data.py
RETAIN_LOGS_PATH = f"saves/eval/tofu_{BASE_MODEL}_{RETAIN_SPLIT}/TOFU_EVAL.json"

# Assuming the setup_data.py was executed before
initial_finetune_eval_output_dir = f"saves/eval/tofu_{BASE_MODEL}_full/evals_{FORGET_SPLIT}"
INITIAL_FINETUNE_SUMMARY_FILE_PATH = os.path.join(initial_finetune_eval_output_dir, "TOFU_SUMMARY.json")


def optuna_setup():
    # Define study name and storage for Optuna
    study_name = "OptiNOVA_LLama-3.2-1B-Instruct_forget10"
    storage_name = "sqlite:///{}.db".format("HP_Opti_NOVA")

    # Create or load the Optuna study. 'minimize' direction is set as our objective is to minimize a combined metric.
    if os.path.exists("sampler_nova.pkl"):
        restored_sampler = pickle.load(open("sampler_nova.pkl", "rb"))
        study_nova = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, sampler=restored_sampler,)
    else:
        study_nova = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,)

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
        logger.info(f"Error during initial finetuning or evaluation: {e}")
        sys.exit(1) # Exit if initial setup fails
    
    return study_nova


# --- Phase 2: Optuna Optimization Loop ---
def objective(trial):
    # Hyperparameters to be optimized by Optuna for the NOVA algorithm
    opt_noise_epochs = trial.suggest_int("noise_epochs", 1, 10)
    opt_noise_lr = trial.suggest_float("noise_lr", 0.001, 0.5, log=True) # Log scale for learning rate
    opt_regularization_term = trial.suggest_float("regularization_term", 0.001, 0.5, log=True) # Log scale for regularization
    opt_impair_gamma = trial.suggest_float("impair_gamma", 0.1, 10.0, log=True) # Log scale for gamma
    opt_repair_alpha = trial.suggest_float("repair_alpha", 0.1, 10.0, log=True) # Log scale for alpha

    # Generate a unique task name for the current trial to store results separately
    trial_task_name = f"nova_trial_{trial.number}_ne{opt_noise_epochs}_nlr{opt_noise_lr:.6f}_reg{opt_regularization_term:.6f}_g{opt_impair_gamma:.2f}_a{opt_repair_alpha:.2f}"
    unlearn_output_dir = f"saves/unlearn/{trial_task_name}"
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
    # eval_command = [
    #     "python", "src/eval.py",
    #     "experiment=eval/tofu/default.yaml", # Use the default TOFU evaluation config
    #     f"forget_split={FORGET_SPLIT}",
    #     f"holdout_split={HOLDOUT_SPLIT}",
    #     f"model={BASE_MODEL}",
    #     f"task_name={trial_task_name}",
    #     f"model.model_args.pretrained_model_name_or_path={unlearn_output_dir}", # Path to the unlearned model from training
    #     f"paths.output_dir={eval_output_dir}", # Set dynamic output path for evaluation logs
    #     f"retain_logs_path={RETAIN_LOGS_PATH}" # Path to reference retain logs for metrics like forget_quality
    # ]

    logger.info(f"Running train command for trial {trial.number}: {' '.join(train_command)}")
    try:
        # Execute the training command
        train_process = subprocess.run(train_command, check=False, capture_output=True, text=True)
        if train_process.returncode != 0:
            logger.info(f"Train command failed for trial {trial.number}. STDOUT:\n{train_process.stdout}\nSTDERR:\n{train_process.stderr}")
            raise RuntimeError("Training process failed")

        logger.info(f"Running eval command for trial {trial.number}: {' '.join(eval_command)}")
        # Execute the evaluation command
        eval_process = subprocess.run(eval_command, check=False, capture_output=True, text=True)
        if eval_process.returncode != 0:
            logger.info(f"Eval command failed for trial {trial.number}. STDOUT:\n{eval_process.stdout}\nSTDERR:\n{eval_process.stderr}")
            raise RuntimeError("Evaluation process failed")

        # Read the evaluation summary results
        if not os.path.exists(summary_file_path):
            raise FileNotFoundError(f"Summary file not found: {summary_file_path}")

        with open(summary_file_path, 'r') as f:
            metrics = json.load(f)

        # Extract the desired metrics from the summary
        forget_qa_prob = metrics.get("forget_Q_A_Prob")
        model_utility = metrics.get("model_utility")

        if forget_qa_prob is None or model_utility is None:
            raise ValueError("Required metrics (forget_Q_A_Prob or model_utility) not found in summary file.")

        if not os.path.exists(INITIAL_FINETUNE_SUMMARY_FILE_PATH):
            raise FileNotFoundError(f"Initial finetune summary file not found: {INITIAL_FINETUNE_SUMMARY_FILE_PATH}")
        with open(INITIAL_FINETUNE_SUMMARY_FILE_PATH, 'r') as f:
            initial_metrics = json.load(f)
        baseline_model_utility = initial_metrics.get("model_utility")

        # Define the objective value to minimize: (forget_Q_A_Prob - model_utility)
        # Lower forget_Q_A_Prob is better (more unlearning), higher model_utility is better (more utility retained).
        # So, minimizing this difference encourages both.
        # Calculate the objective value with the new function
        objective_value = forget_qa_prob + math.abs(baseline_model_utility - model_utility)

        logger.info(f"Trial {trial.number} completed. forget_Q_A_Prob: {forget_qa_prob}, model_utility: {model_utility}, Objective: {objective_value}")
        return objective_value

    except Exception as e:
        logger.info(f"Trial {trial.number} failed due to an error: {e}")
        # Return a very large value to penalize trials that fail or encounter errors
        return float('inf')

def main():
    # Call optuna_setup to initialize the study and get baseline metrics
    study_nova = optuna_setup()

    # Start Trials
    study_nova.optimize(objective, n_trials=1)

    # Save the Optuna sampler state for resuming the study later if needed
    with open("sampler_nova.pkl", "wb") as fout:
        pickle.dump(study_nova.sampler, fout)

if __name__ == "__main__":
    logger.info("Starting the Hyperparameter Tuning")
    main()
