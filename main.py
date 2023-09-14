import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from AlphaZero.Network.trainer import Trainer
from AlphaZero.constants import SAMPLE_ARGS as args_
from AlphaZero.utils import optuna_parameter_search, DotDict
import argparse


def main_alpha_zero_find_hyperparameters():
    parser_ = argparse.ArgumentParser()
    parser_.add_argument("-s", "--storage", help="The storage string to use.")
    parser_.add_argument("-n", "--study_name", help="The name of the optuna study to use.")
    parser_.add_argument("-t", "--n_trials", help="The number of trials to run.")
    parser_.add_argument("-i", "--init_net_path", help="The path to the initial network.")

    args_ = parser_.parse_args()
    storage = args_.storage
    study_name = args_.study_name
    n_trials = int(args_.n_trials)
    init_net_path = args_.init_net_path

    optuna_parameter_search(n_trials=n_trials, init_net_path=init_net_path,
                            storage=storage, study_name=study_name)


def main_alpha_zero():
    for file_name in os.listdir("Logs/AlphaZero"):
        os.remove(f"Logs/AlphaZero/{file_name}")

    args = DotDict(args_)

    # !!! Use found args !!!

    trainer = Trainer.create(args)
    trainer.train()


if __name__ == "__main__":
    # main_alpha_zero()
    main_alpha_zero_find_hyperparameters()
