from abc import ABC, abstractmethod

import numpy as np
import torch as th


class SearchTree(ABC):

    @abstractmethod
    def play_one_game(self, network, device: th.device) -> tuple[list, int, int, int]:
        """
        Performs one game of the algorithm
        """
        pass

    @abstractmethod
    def search(self, network, state: np.ndarray, current_player: int or None, device: th.device,
               tau: float or None = None):
        """
        Performs MCTS search for given number of simulations.
        """
        pass

    @abstractmethod
    def make_fresh_instance(self):
        """
        Return new instance of this class.
        """
        pass

    @abstractmethod
    def step_root(self, action: int or None):
        """
        Steps the root node to the given action.
        """
        pass
