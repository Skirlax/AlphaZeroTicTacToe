# import CpSelfPlay
import numpy as np
import torch as th
from AlphaZero.MCTS.search_tree import McSearchTree
from AlphaZero.utils import DotDict,build_net_from_args
from AlphaZero.constants import SAMPLE_ARGS as _args
from Game.tictactoe_game import TicTacToeGameManager

state = np.zeros((10, 10))
# sample_args = {
#     "board_size": 10,
#     "num_simulations": 1230,
#     "tau": 1.0,
#     "c": 1.0,
#     "num_to_win": 5
# }
# data = th.load("/home/skyr/Downloads/improved_net_3.pth")
# net = bu
sample_args = _args
sample_args.pop("checkpoint_dir")
sample_args.pop("max_depth")
sample_args["num_simulations"] = 1317


res = CpSelfPlay.CmctsSearch(state, 1, sample_args["tau"], sample_args,
                             "/home/skyr/PycharmProjects/AlphaZeroTicTacToe/traced.pt")
print(res)

network = build_net_from_args(sample_args,th.device("cuda"))
network.eval()

tree = McSearchTree(TicTacToeGameManager(10, headless=True, num_to_win=5), DotDict(sample_args))
res = tree.search(network,state,1,th.device("cuda"))
print(res)
