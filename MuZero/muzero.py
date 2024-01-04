import os
import sys

sys.path.append(os.path.abspath("."))
from typing import Type

import torch as th

from AlphaZero.Arena.players import NetPlayer
from AlphaZero.Network.trainer import Trainer
from AlphaZero.constants import SAMPLE_MZ_ARGS
from AlphaZero.utils import DotDict
from Game.asteroids import Asteroids
from General.mz_game import MuZeroGame
from General.network import GeneralNetwork
from MuZero.MZ_Arena.arena import MzArena
from MuZero.MZ_MCTS.mz_search_tree import MuZeroSearchTree
from MuZero.Network.networks import MuZeroNet


class MuZero:
    """
    Class for managing the training and creation of a MuZero model.

    Attributes:
        game_manager (MuZeroGame): The game manager instance.
        net (GeneralNetwork): The neural network used by MuZero for prediction.
        trainer (Trainer): The trainer object responsible for training the model.
        device (torch.device): The device (CPU or CUDA) used for computations.

    Methods:
        __init__(self, game_manager: MuZeroGame)
            Initializes a new instance of the MuZero class.

        create_new(self, args: dict, network_class: Type[GeneralNetwork], headless: bool = True,
                   checkpointer_verbose: bool = False)
            Creates a new MuZero model using the specified arguments.

        train(self)
            Trains the MuZero model.
    """

    def __init__(self, game_manager: MuZeroGame):
        self.game_manager = game_manager
        self.net = None
        self.trainer = None
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

    def create_new(self, args: dict, network_class: Type[GeneralNetwork], headless: bool = True,
                   checkpointer_verbose: bool = False):
        args["net_action_size"] = int(self.game_manager.get_num_actions())
        network = network_class.make_from_args(args).to(self.device)
        tree = MuZeroSearchTree(self.game_manager.make_fresh_instance(), args)
        net_player = NetPlayer(self.game_manager.make_fresh_instance(),
                               **{"network": network, "monte_carlo_tree_search": tree})
        args = DotDict(args)
        args.self_play_games = 300
        args.epochs = 500
        args.lr = 0.0032485504583772953
        args.tau = 1.0
        args.c = 1
        args.arena_tau = 0.04139160592420218
        arena = MzArena(self.game_manager.make_fresh_instance(), args, self.device)
        self.trainer = Trainer.create(args, self.game_manager.make_fresh_instance(), network, tree, net_player,
                                      headless=headless,
                                      checkpointer_verbose=checkpointer_verbose, arena_override=arena)
        self.net = self.trainer.get_network()

    def train(self):
        self.trainer.train()


if __name__ == "__main__":
    game = Asteroids()
    mz = MuZero(game)
    mz.create_new(SAMPLE_MZ_ARGS, MuZeroNet)
    mz.train()
