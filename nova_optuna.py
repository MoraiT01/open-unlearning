import sys
import os
import subprocess
import json
import math
import time
import logging
import optuna

from hpsearch_setup import Config, create_study_with_storage

# --- 2. Modular Logging Setup ---
def setup_logging():
    """Configures the logger for the HPTUNER."""
    logger = logging.getLogger("HPTUNER")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')

    file_handler = logging.FileHandler(Config.LOG_FILE, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

logger = setup_logging()


# --- 3. Helper Functions ---
def get_initial_metrics(summary_file_path):
    """Loads and returns baseline metrics from a summary file."""
    try:
        if not os.path.exists(summary_file_path):
            raise FileNotFoundError(f"Initial finetune summary file not found: {summary_file_path}")
        with open(summary_file_path, 'r') as f:
            initial_metrics = json.load(f)
        return initial_metrics
    except Exception as e:
        logger.error(f"Error loading initial metrics from {summary_file_path}: {e}")
        sys.exit(1)

def scale_to_0_1(original_value, x1, y1):
    """Scales a value from the interval [x1, y1] to [0, 1]."""
    if y1 <= x1:
        raise ValueError("y1 must be greater than x1 for scaling.")
    scaled_value = (original_value - x1) / (y1 - x1)
    return scaled_value

# --- 4. Refined Objective Function ---
def objective(trial):
    """The main optimization objective function for Optuna."""
    # Hyperparameters from the predefined grid.
    # The GridSampler will override these suggestions with values from the grid.
    opt_noise_epochs = trial.suggest_int("noise_epochs", 1, 100)
    opt_noise_lr = trial.suggest_float("noise_lr", 1e-10, 1.0, log=True)
    opt_regularization_term = trial.suggest_float("regularization_term", 1e-8, 1.0, log=True)
    opt_alpha = trial.suggest_float("alpha", 1e-10, 1.0, log=True)
    opt_sign = trial.suggest_categorical("sign", [-1, 1])
    opt_soft_targets = trial.suggest_categorical("soft_targets", [0, 1])

    # Dynamic path generation
    unlearn_output_dir_base = "saves/unlearn"
    unlearn_output_dir = os.path.join(unlearn_output_dir_base, Config.BASE_MODEL, Config.FORGET_SPLIT, f"nova_trial_{trial.number}")
    eval_output_dir = os.path.join(unlearn_output_dir, "evals")
    summary_file_path = os.path.join(eval_output_dir, "TOFU_SUMMARY.json")

    os.makedirs(unlearn_output_dir, exist_ok=True)
    os.makedirs(eval_output_dir, exist_ok=True)
    s_target = True if int(opt_soft_targets)==int(1) else False
    try:
        # Construct and run the training command
        train_command = [
            "python", "src/train.py",
            "--config-name=unlearn.yaml",
            "experiment=unlearn/tofu/default",
            f"trainer=NOVA",
            f"task_name=nova_trial_{trial.number}",
            f"model={Config.BASE_MODEL}",
            f"forget_split={Config.FORGET_SPLIT}",
            f"retain_split={Config.RETAIN_SPLIT}",
            f"retain_logs_path={Config.RETAIN_LOGS_PATH}",
            f"trainer.method_args.noise_epochs={opt_noise_epochs}",
            f"trainer.method_args.noise_lr={opt_noise_lr}",
            f"trainer.method_args.regularization_term={opt_regularization_term}",
            f"trainer.method_args.alpha={opt_alpha}",
            f"trainer.method_args.sign={opt_sign}",
            f"trainer.method_args.soft_target={s_target}",
            f"paths.output_dir={unlearn_output_dir}",
        ]
        logger.info(f"Starting Train Trial {trial.number}: {' '.join(train_command)}")
        start_time = time.time()
        subprocess.run(train_command, check=True, capture_output=True, text=True)
        training_time = time.time() - start_time

        # Construct and run the evaluation command
        eval_command = [
            "python", "src/eval.py",
            "experiment=eval/tofu/default.yaml",
            f"forget_split={Config.FORGET_SPLIT}",
            f"holdout_split={Config.HOLDOUT_SPLIT}",
            f"model={Config.BASE_MODEL}",
            f"task_name=nova_trial_{trial.number}",
            f"model.model_args.pretrained_model_name_or_path={unlearn_output_dir}",
            f"paths.output_dir={eval_output_dir}",
            f"retain_logs_path={Config.RETAIN_LOGS_PATH}"
        ]
        logger.info(f"Starting Eval Trial {trial.number}: {' '.join(eval_command)}")
        subprocess.run(eval_command, check=True, capture_output=True, text=True)

        # Load initial and trial metrics
        initial_metrics = get_initial_metrics(Config.INITIAL_FINETUNE_SUMMARY_FILE_PATH)
        with open(summary_file_path, 'r') as f:
            metrics = json.load(f)

        # Calculate objective value
        forget_quality = metrics["forget_quality"]
        model_utility = metrics["model_utility"]
        baseline_forget_quality = initial_metrics["forget_quality"]
        baseline_model_utility = initial_metrics["model_utility"]
        
        delta_model_utility = abs(baseline_model_utility - model_utility)
        log_baseline_forget_quality = math.log10(baseline_forget_quality)
        log_forget_quality = math.log10(forget_quality)

        scaled_forget_quality = scale_to_0_1(log_forget_quality, log_baseline_forget_quality, 0)
        scaled_delta_model_utility = scale_to_0_1(delta_model_utility, 0, max(baseline_model_utility, 1 - baseline_model_utility))
        
        objective_value = scaled_forget_quality

        logger.info(f"Trial {trial.number} complete. Forget Q: {scaled_forget_quality:.4f}, Delta Utility: {scaled_delta_model_utility:.4f}, Objective: {objective_value:.4f}")

        # Store user attributes
        trial.set_user_attr("Training Time (min)", training_time)
        trial.set_user_attr("Forget Quality (log10-scale)", log_forget_quality)
        trial.set_user_attr("Model Utility", model_utility)

        # Cleanup
        if not Config.KEEP_MODEL_TENSORS:
            model_tensors_file_path = os.path.join(unlearn_output_dir, "model.safetensors")
            if os.path.exists(model_tensors_file_path):
                os.remove(model_tensors_file_path)
                logger.info(f"Deleted {model_tensors_file_path} for trial {trial.number}.")

        return objective_value

    except (subprocess.CalledProcessError, FileNotFoundError, KeyError, ValueError) as e:
        logger.error(f"Trial {trial.number} failed with an error: {e}")
        # Mark the trial as failed and return a poor score
        return float('-inf')


def run_worker():
    """A single worker function for parallel optimization."""
    study = create_study_with_storage()
    # The GridSampler will automatically stop when all trials in the grid are completed.
    study.optimize(objective, n_trials=Config.NUM_TRIALS)


if __name__ == "__main__":
    logger.info("Starting the Grid Search process for Optuna.")

    run_worker()
    
    # After all workers are finished, load the final study to print results
    final_study = optuna.load_study(
        study_name=Config.STUDY_NAME,
        storage=Config.STORAGE_NAME
    )
    
    logger.info(f"Total number of trials: {len(final_study.trials)}")
    logger.info(f"Best trial: {final_study.best_trial.value}")
    logger.info(f"Best parameters: {final_study.best_trial.params}")