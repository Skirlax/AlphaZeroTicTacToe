from abc import ABC, abstractmethod

import numpy as np


class Game(ABC):
    """
    Make your custom game extend this class and implement all the methods.
    """

    def game_result(self, player, board) -> float | None:
        """
        Returns the result of the game from the perspective of the supplied player.
        :param player: The player to check for (1 or -1).
        :param board: The board to check on. If None, the current board is used.
        :return: The game result. None when the game is not over yet.
        """
        if self.check_win(player, board=board):
            return 1.0
        if self.check_win(-player, board=board):
            return -1.0
        if self.is_board_full(board=board):
            return 1e-4
        return None

    @staticmethod
    def adjust_probabilities(action_probs: dict, tau=1.0) -> dict:
        """
        Selects a move from the action probabilities using either greedy or stochastic policy.
        The stochastic policy uses the tau parameter to adjust the probabilities. This is based on the
        temperature parameter in DeepMind's AlphaZero paper.

        :param action_probs: A dictionary containing the action probabilities in the form of {action_index: probability}.
        :param tau: The temperature parameter. 0 for greedy, >0 for stochastic.
        :return: The selected move as an integer (index).
        """
        if tau == 0:  # select greedy
            vals = [x for x in action_probs.values()]
            max_idx = vals.index(max(vals))
            probs = [0 for _ in range(len(vals))]
            probs[max_idx] = 1
            return dict(zip(action_probs.keys(), probs))
        # select stochastic
        moves, probabilities = zip(*action_probs.items())
        adjusted_probs = [prob ** (1 / tau) for prob in probabilities]
        adjusted_probs_sum = sum(adjusted_probs)
        normalized_probs = [prob / adjusted_probs_sum for prob in adjusted_probs]
        return dict(zip(moves, normalized_probs))

    @staticmethod
    def select_move(action_probs: dict):
        moves, probs = zip(*action_probs.items())
        return np.random.choice(moves, p=probs)

    @abstractmethod
    def make_fresh_instance(self):
        """
        Returns a fresh instance of the game (not a copy).
        """
        pass

    @abstractmethod
    def check_win(self, player: int, board: np.ndarray) -> bool:
        """
        Checks if the current player 'player' won on the 'board'. If so returns true, else false.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_canonical_form(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Returns the canonical form of the board.
        In this case, that is a board where the current player is always player 1.
        You probably want: return board * player
        """
        pass

    @abstractmethod
    def get_next_state(self, board: np.ndarray, board_index: tuple, player: int) -> np.ndarray:
        """
        This method plays move given by 'board_index' as 'player' on the given 'board'
        and returns the updated board.
        """
        pass

    @abstractmethod
    def is_board_full(self, board: np.ndarray) -> bool:
        """
        Checks if the given board is completely full. Returns true if so, false other otherwise,
        """
        pass

    @abstractmethod
    def get_valid_moves(self, board: np.ndarray, player: int or None) -> np.ndarray:
        """
        Returns a list of valid moves for the given player on the given board.
        If valid moves are independent of player, use None for player.
        """
        pass

    @abstractmethod
    def get_board(self) -> np.ndarray:
        """
        Returns a copy of the current board.
        """
        pass

    def eval_board(self, board: np.ndarray, player: int):
        """
        Optional. Evaluates the board from the perspective of the given player.
        This is used for minimax, override this method if you want to use it.
        """
        pass
