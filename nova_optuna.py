import logging
import sys
import pickle
import optuna
import os


# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "OptiNOVA"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format("HP_Opti")

if os.path.exists("sampler_nova.pkl"):
    restored_sampler = pickle.load(open("sampler_nova.pkl", "rb"))
    study_nova = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, sampler=restored_sampler)
else:
    study_nova = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,)


def objective(trial):

    opt_noise_epochs = trial.suggest_int("Noise_Epochs", 1, 10)
    opt_noise_lr = trial.suggest_float("Noise_LR", 0, 1)
    opt_regularization_term = trial.suggest_float("Regularization", 0.001, 0.5)
    opt_impair_gamma = trial.suggest_float("Gamma", 0.001, 1.)
    opt_repair_alpha = trial.suggest_float("Alpha", 0.001, 1.)


    # # These are the 
    # {
    #     "extraction_strength": 0.03250892997513522,
    #     "forget_Q_A_Prob": 4.518372597802884e-39,
    #     "forget_Q_A_ROUGE": 0.00033982174859190984,
    #     "model_utility": 0.0,
    #     "privleak": 16.478124996704384
    # }

    return 0

study_nova.optimize(objective, n_trials=50)

# Save the sampler with pickle to be loaded later.
with open("sampler_nova.pkl", "wb") as fout:
    pickle.dump(study_nova.sampler, fout)