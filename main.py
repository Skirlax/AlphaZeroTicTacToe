import os

# os.environ['OMP_NUM_THREADS'] = '5'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from AlphaZero.Network.trainer import Trainer

from AlphaZero.constants import SAMPLE_ARGS as args_
from AlphaZero.constants import TRAINED_NET_ARGS as _args
from AlphaZero.utils import optuna_parameter_search, DotDict, make_net_from_checkpoint,build_net_from_args
import argparse
from AlphaZero.Arena.players import NetPlayer, HumanPlayer, MinimaxPlayer
from AlphaZero.MCTS.search_tree import McSearchTree
from Game.tictactoe_game import TicTacToeGameManager
from AlphaZero.Arena.arena import Arena

import torch as th


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

    args.num_simulations = 1317
    args.self_play_games = 145
    args.epochs = 320
    args.lr = 0.0032485504583772953
    # args.tau = 0.8243290871234756
    args.tau = 1.0
    args.c = 1.15
    args.arena_tau = 0.04139160592420218
    # args.arena_tau = args.tau
    # args.log_epsilon = 1.4165210108199043e-08

    args.num_iters = 50

    # trainer = Trainer.create(args)
    trainer = Trainer.create(args)
    trainer.train()
    trainer.save_latest("Checkpoints/AlphaZero/latest.pth")


def play():
    args = DotDict(args_)
    # args.num_to_win = 3
    # args.board_size = 5
    # args.net_action_size = args.board_size ** 2
    net = make_net_from_checkpoint("/home/skyr/Downloads/improved_net_1.pth", args)
    net.eval()
    manager = TicTacToeGameManager(8, headless=False, num_to_win=5)
    search_tree = McSearchTree(manager, args)
    # p1 = TrainingNetPlayer(net,manager,args)
    p1 = NetPlayer(net, search_tree, manager)
    opp_data = th.load("/home/skyr/Downloads/temp_net_119.pth")
    opp_net = build_net_from_args(args, th.device("cuda"))
    opp_net.load_state_dict(opp_data)
    manager2 = TicTacToeGameManager(8, headless=False, num_to_win=5)
    search_tree2 = McSearchTree(manager2, args)
    # p1 = RandomPlayer(manager)
    # p2 = HumanPlayer(manager)
    p2 = NetPlayer(opp_net,search_tree2,manager2)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    arena = Arena(manager, args, device)
    net.eval()
    opp_net.eval()
    net_wins, human_wins, draws = arena.pit(p1, p2, num_games_to_play=10, num_mc_simulations=1317, one_player=False,
                                            start_player=1,debug=True)
    print(f"Net wins: {net_wins}, Human wins: {human_wins}, Draws: {draws}")


def human_minimax_play():
    args = DotDict(_args)
    manager = TicTacToeGameManager(8, headless=False, num_to_win=5)
    p1 = HumanPlayer(manager)
    p2 = MinimaxPlayer(manager, evaluate_fn=manager.eval_board)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    arena = Arena(manager, args, device)
    net_wins, human_wins, draws = arena.pit(p1, p2, num_games_to_play=10, num_mc_simulations=1317, one_player=True,
                                            start_player=1, add_to_kwargs={"depth": 3, "player": -1})


if __name__ == "__main__":
    human_minimax_play()
    # play()
    #
    # main_alpha_zero()
    # main_alpha_zero_find_hyperparameters()
