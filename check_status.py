import optuna
import sys

from hpsearch_setup import Config

# Set these to match your Optuna study
STUDY_NAME = Config.STUDY_NAME
STORAGE_NAME = Config.STORAGE_NAME
GRID_SEARCH_SPACE = Config.GRID_SEARCH_SPACE

try:
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME
    )
    
    # Count how many trials are not in a completed or failed state
    # Firstly, let's see how many it should be
    individual_lengths = [len(value) for _, value in GRID_SEARCH_SPACE.items()]
    res = 1
    for val in individual_lengths:
        res = res * val
    pending_trials = res

    # Secondly, how many have been taken care of so far
    all_trials = study.get_trials()
    for trial in all_trials:
        # Different States: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.TrialState.html#optuna.trial.TrialState
        if trial.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.WAITING]:
            pending_trials -= 1

    print(pending_trials)

except optuna.exceptions.StorageInternalError:
    print("0") # If the study doesn't exist, there are no trials to run.
except Exception as e:
    # A generic error handling for any other DB issues.
    print("-1")
    sys.exit(1)