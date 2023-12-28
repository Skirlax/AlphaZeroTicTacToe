from typing import Type

import torch as th
from AlphaZero.constants import SAMPLE_MZ_ARGS
from AlphaZero.Arena.players import NetPlayer
from AlphaZero.Network.trainer import Trainer
from AlphaZero.utils import DotDict
from General.mz_game import MuZeroGame
from General.network import GeneralNetwork
from MuZero.MZ_MCTS.mz_search_tree import MuZeroSearchTree
from Game.asteroids import Asteroids
from MuZero.Network.networks import MuZeroNet

class MuZero:
    def __init__(self, game_manager: MuZeroGame):
        self.game_manager = game_manager
        self.net = None
        self.trainer = None
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

    def create_new(self, args: dict, network_class: Type[GeneralNetwork], headless: bool = True,
                   checkpointer_verbose: bool = True):
        args["net_action_size"] = int(self.game_manager.get_num_actions())
        network = network_class.make_from_args(args).to(self.device)
        tree = MuZeroSearchTree(self.game_manager.make_fresh_instance(), args)
        net_player = NetPlayer(self.game_manager.make_fresh_instance(),
                               **{"network": network, "monte_carlo_tree_search": tree})
        args = DotDict(args)
        self.trainer = Trainer.create(args, self.game_manager.make_fresh_instance(), network,tree, net_player,
                                      headless=headless,
                                      checkpointer_verbose=checkpointer_verbose)
        self.net = self.trainer.get_network()

    def train(self):
        self.trainer.train()



if __name__ == "__main__":
    game = Asteroids()
    mz = MuZero(game)
    mz.create_new(SAMPLE_MZ_ARGS,MuZeroNet)
    mz.train()
