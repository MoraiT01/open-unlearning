import os

import optuna
from optuna.samplers import GridSampler

# --- 1. Centralized Configuration Class ---
class Config:
    """A class to hold all configuration constants for the experiment."""
    # Experiment settings
    BASE_MODEL = "Llama-3.2-1B-Instruct"
    FORGET_SPLIT = "forget10"
    RETAIN_SPLIT = "retain90"
    HOLDOUT_SPLIT = "holdout10"

    # Optuna and Unlearning settings
    MAXIMIZE_FORGETTING = True
    STUDY_NAME = f"GridSearch_NOVA_{BASE_MODEL}_{FORGET_SPLIT}"
    STORAGE_NAME = "sqlite:///HP_GridSearch_NOVA.db"
    NUM_TRIALS = 1

    # Define the Grid Search Space
    # TODO -> Set the Search Space
    GRID_SEARCH_SPACE = {
        "noise_epochs": [5, 10],
        "noise_lr": [1e-4],
        "regularization_term": [1e-3],
        "impair_gamma": [5.0],
        "repair_alpha": [1.0],
        "soft_targets": [False],
    }

    # Paths (assuming data is downloaded and structured)
    INITIAL_FINETUNE_EVAL_OUTPUT_DIR = f"saves/eval/tofu_{BASE_MODEL}_full/evals_{FORGET_SPLIT}"
    INITIAL_FINETUNE_SUMMARY_FILE_PATH = os.path.join(INITIAL_FINETUNE_EVAL_OUTPUT_DIR, "TOFU_SUMMARY.json")
    RETAIN_LOGS_PATH = f"saves/eval/tofu_{BASE_MODEL}_{RETAIN_SPLIT}/TOFU_EVAL.json"

    # Logging
    LOG_FILE = f"logs/gridsearch_{BASE_MODEL}_{FORGET_SPLIT}.log"
    # Cleanup
    KEEP_MODEL_TENSORS = False
    
    # Environment variables
    os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


def create_study_with_storage():
    """Creates or loads an Optuna study using a persistent storage and GridSampler."""
    
    # We remove the try/except block.
    # The 'load_if_exists=True' parameter handles the logic of loading vs. creating.
    grid_sampler = GridSampler(Config.GRID_SEARCH_SPACE)
    study = optuna.create_study(
        study_name=Config.STUDY_NAME,
        storage=Config.STORAGE_NAME,
        direction="maximize",
        sampler=grid_sampler,
        load_if_exists=True  # This is the key
    )
    
    return study


if __name__ == "__main__":
    print(f"Loading or Creating Study: {Config.STUDY_NAME}")
    create_study_with_storage()