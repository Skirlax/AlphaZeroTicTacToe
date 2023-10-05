from abc import ABC, abstractmethod

import numpy as np

from AlphaZero.MCTS.search_tree import McSearchTree
from AlphaZero.Network.nnet import TicTacToeNet
from Game.game import GameManager


class Player(ABC):

    @abstractmethod
    def choose_move(self, board: np.ndarray, **kwargs) -> tuple[int, int]:
        pass


class RandomPlayer(Player):
    def __init__(self, game_manager: GameManager):
        self.game_manager = game_manager
        self.name = "RandomPlayer"

    def choose_move(self, board: np.ndarray, **kwargs) -> tuple[int, int]:
        move = self.game_manager.get_random_valid_action(board)
        return tuple(move)


class NetPlayer(Player):
    def __init__(self, network: TicTacToeNet, mc_tree_search: McSearchTree, game_manager: GameManager):
        self.game_manager = game_manager
        self.network = network
        self.monte_carlo_tree_search = mc_tree_search
        self.name = "NetworkPlayer"

    def choose_move(self, board: np.ndarray, **kwargs) -> tuple[int, int]:
        try:
            current_player = kwargs["current_player"]
            device = kwargs["device"]
            tau = kwargs["tau"]
        except KeyError:
            raise KeyError("Missing keyword argument. Please supply kwargs: current_player, device, "
                           "tau")

        pi, _ = self.monte_carlo_tree_search.search(self.network, board, current_player, device,tau=tau)
        move = self.game_manager.select_move(pi)
        self.monte_carlo_tree_search.step_root(None)
        return self.game_manager.network_to_board(move)


class HumanPlayer(Player):
    def __init__(self,game_manager: GameManager):
        self.name = "HumanPlayer"
        self.game_manager = game_manager

    def choose_move(self, board: np.ndarray, **kwargs) -> tuple[int, int]:
        if self.game_manager.headless:
            raise RuntimeError("Cannot play with a human player in headless mode.")
        move = self.game_manager.get_human_input(board)
        return move

