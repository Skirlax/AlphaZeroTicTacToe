from abc import ABC, abstractmethod

import numpy as np
import torch as th


class MuZeroGame(ABC):

    @abstractmethod
    def get_next_state(self, action: int, player: int or None) -> (
            np.ndarray or th.Tensor, int, bool):
        """
        Given a game state and an action return the next state. If player is None, one player is assumed.
        :param state: The current state of the game.
        :param action: The action to be taken.
        :param player: The player taking the action.
        :return: The next state, the reward and a boolean indicating if the game is done.
        """
        pass

    @abstractmethod
    def reset(self) -> np.ndarray or th.Tensor:
        """
        Resets the game and returns the initial board.
        """
        pass

    @abstractmethod
    def get_noop(self) -> int:
        """
        Returns the noop action.
        """
        pass

    @abstractmethod
    def get_num_actions(self) -> int:
        """
        Returns the number of actions possible in the current environment.
        """
        pass

    @abstractmethod
    def game_result(self, player: int or None) -> bool or None:
        """
        Returns true if this environment is done, false otherwise. If this environment doesn't have a clear terminal
        state, return None. If player is None, one player is assumed.
        """
        pass

    @abstractmethod
    def make_fresh_instance(self):
        """
        Return fresh instance of this game manager.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Render the GUI of the game.
        """
        pass

    @abstractmethod
    def frame_skip_wrapper(self, action: int, player: int or None, frame_skip: int = 4) -> (
            np.ndarray or th.Tensor, int, bool):
        """
        Wrapper for the frame skip method. This method should be used instead of get_next_state.
        """
        pass

    @staticmethod
    @abstractmethod
    def select_move(action_probs: dict):
        pass

    def get_random_valid_action(self,board: np.ndarray):
        pass
