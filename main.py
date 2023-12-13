import torch as th
from AlphaZero.Arena.arena import Arena
from Game.game import GameManager
from AlphaZero.MCTS.search_tree import McSearchTree
from AlphaZero.Arena.players import NetPlayer, HumanPlayer, RandomPlayer, MinimaxPlayer
import argparse
from AlphaZero.utils import optuna_parameter_search, DotDict, make_net_from_checkpoint
from AlphaZero.constants import TRAINED_NET_ARGS as _args
from AlphaZero.constants import SAMPLE_ARGS as args_
from AlphaZero.Network.trainer import Trainer
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


def main_alpha_zero_find_hyperparameters():
    parser_ = argparse.ArgumentParser()
    parser_.add_argument("-s", "--storage", help="The storage string to use.")
    parser_.add_argument("-n", "--study_name",
                         help="The name of the optuna study to use.")
    parser_.add_argument("-t", "--n_trials",
                         help="The number of trials to run.")
    parser_.add_argument("-i", "--init_net_path",
                         help="The path to the initial network.")

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

    args.num_simulations = 1317
    args.self_play_games = 145
    args.epochs = 320
    args.lr = 0.00032485504583772953
    # args.tau = 0.8243290871234756
    args.tau = 1.0
    args.c = 1.7409658274805089
    args.arena_tau = 0  # 0.04139160592420218
    # args.arena_tau = args.tau
    # args.log_epsilon = 1.4165210108199043e-08

    args.num_iters = 50

    trainer = Trainer.create(args)
    trainer.train()
    trainer.save_latest("Checkpoints/AlphaZero/latest.pth")


def play():
    args = DotDict(_args)
    net = make_net_from_checkpoint(
        "Nets/FinalNets/5x5_3/latest_trained_net.pth", args)
    net.eval()
    manager = GameManager(5, headless=False, num_to_win=3)
    search_tree = McSearchTree(manager, args)
    p1 = NetPlayer(net, search_tree, manager)
    # p1 = RandomPlayer(manager)
    p2 = HumanPlayer(manager)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    arena = Arena(manager, args, device)
    net_wins, human_wins, draws = arena.pit(p1, p2, num_games_to_play=10, num_mc_simulations=1317, one_player=True,
                                            start_player=1)
    print(f"Net wins: {net_wins}, Human wins: {human_wins}, Draws: {draws}")


def human_minimax_play():
    args = DotDict(_args)
    manager = GameManager(3, headless=False, num_to_win=3)
    p1 = HumanPlayer(manager)
    p2 = MinimaxPlayer(manager, evaluate_fn=manager.eval_board)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    arena = Arena(manager, args, device)
    net_wins, human_wins, draws = arena.pit(p1, p2, num_games_to_play=10, num_mc_simulations=1317, one_player=True,
                                            start_player=1, add_to_kwargs={"depth": 5, "player": -1})


if __name__ == "__main__":
    # human_minimax_play()
    # play()
    #
    main_alpha_zero()
    # main_alpha_zero_find_hyperparameters()
