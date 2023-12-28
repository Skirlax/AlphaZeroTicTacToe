import sys
from typing import Type

import torch as th

from AlphaZero.Arena.arena import Arena
from AlphaZero.Arena.players import NetPlayer
from AlphaZero.MCTS.az_search_tree import McSearchTree
from AlphaZero.Network.trainer import Trainer
from AlphaZero.utils import DotDict
from General.az_game import Game
from General.network import GeneralNetwork


class AlphaZero:
    def __init__(self, game_instance: Game):
        self.trainer = None
        self.net = None
        self.game = game_instance
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

    def create_new(self, args: dict, network_class: Type[GeneralNetwork], headless: bool = True,
                   checkpointer_verbose: bool = False):
        network = network_class.make_from_args(args)
        tree = McSearchTree(self.game.make_fresh_instance(), args)
        net_player = NetPlayer(self.game.make_fresh_instance(), **{"network": network, "monte_carlo_tree_search": tree})
        args = DotDict(args)
        self.trainer = Trainer.create(args, self.game, tree, net_player, headless=headless,
                                      checkpointer_verbose=checkpointer_verbose)
        self.net = self.trainer.get_network()

    def load_checkpoint(self, path: str, checkpoint_dir: str, headless: bool = True,
                        checkpointer_verbose: bool = False):
        self.trainer = Trainer.from_checkpoint(path, checkpoint_dir, self.game, headless=headless,
                                               checkpointer_verbose=checkpointer_verbose)
        self.net = self.trainer.get_network()

    def train(self):
        self.trainer.train()

    def play(self, p1_name: str, p2_name: str, num_games: int, args: dict, starts: int = 1,
             switch_players: bool = True):
        assert self.net is not None, ("Network is None, can't pit. Make sure you initialize the network with either "
                                      "load_checkpoint or create_new method.")
        args = DotDict(args)
        manager = self.game.make_fresh_instance()
        tree = McSearchTree(manager, args)
        kwargs = {"network": self.net, "monte_carlo_tree_search": tree, "evaluate_fn": manager.eval_board}
        p1 = sys.modules["AlphaZero.Arena.players"].__dict__[p1_name](manager, **kwargs)
        p2 = sys.modules["AlphaZero.Arena.players"].__dict__[p2_name](manager, **kwargs)
        arena_manager = self.game.make_fresh_instance()
        arena_manager.set_headless(False)
        arena = Arena(arena_manager, args, self.device)
        p1_w, p2_w, ds = arena.pit(p1, p2, num_games, args["num_simulations"], one_player=not switch_players,
                                   start_player=starts)
        print(f"Results: Player 1 wins: {p1_w}, Player 2 wins: {p2_w}, Draws: {ds}")
