import random

import numpy as np
import pygame as pg


class GameManager:
    """
    This class is the game manager for the game of Tic Tac Toe and its variants.
    """

    def __init__(self, board_size: int, headless: bool) -> None:
        # TODO: Implement iteration over ALL diagonals to support partial wins (num_to_win < board_size).
        # TODO: Implement the possibility to play over internet using sockets.
        self.player = 1
        self.enemy_player = -1
        self.num_to_win = board_size
        self.board_size = board_size
        self.board = self.initialize_board()
        self.headless = headless
        self.screen = self.create_screen(headless)

    def play(self, player: int, index: tuple) -> None:
        self.board[index] = player

    def initialize_board(self):
        board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        return board

    def get_random_valid_action(self, observations: np.ndarray) -> list:
        valid_moves = self.get_legal_moves_on_observation(observations)
        if len(valid_moves) == 0:
            raise Exception("No valid moves")
        return random.choice(valid_moves)

    def create_screen(self, headless):
        if headless:
            return

        pg.init()
        pg.display.set_caption("Tic Tac Toe")
        board_rect_size = self.board_size * 100
        screen = pg.display.set_mode((board_rect_size, board_rect_size))
        return screen

    def full_iterate_array(self, arr: np.ndarray, func: callable) -> list:
        """
        This function iterates over all rows, columns and the two main diagonals of supplied array,
        applies the supplied function to each of them and returns the results in a list.
        :param arr: a 2D numpy array.
        :param func: a callable function that takes a 1D array as input and returns a result.
        :return: A list of results.
        """
        results = []
        for row in arr:
            results.append(func(row.reshape(-1)))

        results.append("col")
        for col in arr.T:
            results.append(func(col.reshape(-1)))

        results.append("diag_left")
        results.append(func(arr.diagonal().reshape(-1)))
        results.append("diag_right")
        results.append(func(np.fliplr(arr).diagonal().reshape(-1)))
        return results

    def check_full_win(self, player: int, board=None) -> bool:
        """
        This function checks if the supplied player has won the game with a full win (num_to_win == board_size).
        :param player: The player to check for (1 or -1).
        :param board: The board to check on. If None, the current board is used.
        :return: True if the player has won, False otherwise.
        """
        if board is None:
            board = self.get_board()
        matches = self.full_iterate_array(board, lambda part: np.all(part == player))
        for match in matches:
            if not isinstance(match, str) and match:
                return True

        return False

    def check_partial_win(self, player: int, n: int, board=None) -> bool:
        """
        This function checks if the supplied player has won the game with a partial win (num_to_win < board_size).

        :param player: The player to check for (1 or -1).
        :param n: The number of consecutive pieces needed to win.
        :param board: The board to check on. If None, the current board is used.
        :return: True if the player has won, False otherwise.
        """
        raise NotImplementedError("This will be usable when the full_iterate_array function iterates over ALL "
                                  "diagonals.")
        if board is None:
            board = self.get_board()
        matches = self.full_iterate_array(board,
                                          lambda part:
                                          np.convolve((part == player), np.ones(n, dtype=np.int),
                                                      "valid"))

        for match in matches:
            if np.any(match == n):
                return True

        return False

    def check_partial_win_to_index(self, player: int, n: int, board=None) -> dict[tuple, str] | dict:
        """
        This variation of check_partial_win returns the index of the first partial win found.
        The index being the index of the first piece in the winning sequence.
        :param player: The player to check for (1 or -1).
        :param n: The number of consecutive pieces needed to win.
        :param board: The board to check on. If None, the current board is used.
        :return: A dictionary containing the index and the position of the winning sequence.
        """
        if board is None:
            board = self.get_board()
        indices = self.full_iterate_array(board,
                                          lambda part: np.where(
                                              np.convolve((part == player), np.ones(n, dtype=int),
                                                          "valid") == n)[0])

        pos = "row"
        for index in indices:
            if type(index) == str:
                pos = index
                continue
            if len(index) > 0:
                return {"index": np.unravel_index(index[0], shape=self.board.shape), "pos": pos}

        return {}

    def is_board_full(self, board=None) -> bool:
        if board is None:
            board = self.get_board()
        return np.all(board != 0)

    def get_board(self):
        return self.board.copy()

    def reset(self):
        self.board = self.initialize_board()
        return self.board.copy()

    def get_board_size(self):
        return self.board_size

    def pygame_render(self) -> bool:
        if self.headless:
            return False

        self.screen.fill((0, 0, 0))
        self._draw_board()
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == self.player:
                    self._draw_circle(col * 100 + 50, row * 100 + 50)

                elif self.board[row][col] == self.enemy_player:
                    self._draw_cross(col * 100 + 50, row * 100 + 50)

        pg.event.pump()
        pg.display.update()
        return True

    def _draw_circle(self, x, y) -> None:
        if self.headless:
            return
        pg.draw.circle(self.screen, "blue", (x, y), 40, 1)

    def _draw_cross(self, x, y) -> None:
        if self.headless:
            return
        pg.draw.line(self.screen, "red", (x - 40, y - 40), (x + 40, y + 40), 1)
        pg.draw.line(self.screen, "red", (x + 40, y - 40), (x - 40, y + 40), 1)

    def _draw_board(self):
        for x in range(0, self.board_size * 100, 100):
            for y in range(0, self.board_size * 100, 100):
                pg.draw.rect(self.screen, (255, 255, 255), pg.Rect(x, y, 100, 100), 1)

    def is_empty(self, index: tuple) -> bool:
        return self.board[index] == 0

    def get_legal_moves_on_observation(self, observation: np.ndarray) -> list:
        """
        Legal moves are the empty spaces on the board.
        :param observation: A 2D numpy array representing the current state of the game.
        :return: A list of legal moves.
        """
        legal_moves = []
        observation = observation.reshape(self.board_size, self.board_size)
        for row in range(self.board_size):
            for col in range(self.board_size):
                if observation[row][col] == 0:
                    legal_moves.append([row, col])
        return legal_moves

    def pygame_quit(self) -> bool:
        if self.headless:
            return False
        pg.quit()
        return True

    def game_result(self, player, board) -> float | None:
        """
        Returns the result of the game from the perspective of the supplied player.
        :param player: The player to check for (1 or -1).
        :param board: The board to check on. If None, the current board is used.
        :return: The game result. None when the game is not over yet.
        """
        if self.check_full_win(player, board=board):
            return 1.0
        elif self.check_full_win(-player, board=board):
            return -1.0
        elif self.is_board_full(board=board):
            return 1e-4
        else:
            return None

    def network_to_board(self, move):
        """
        Converts an integer move from the network to a board index.
        :param move: An integer move selected from the network probabilities.
        :return: A tuple representing the board index (int,int).
        """
        return np.unravel_index(move, self.board.shape)

    @staticmethod
    def get_canonical_form(board, player) -> np.ndarray:
        return board * player

    @staticmethod
    def get_next_state(board, board_index, player) -> np.ndarray:
        board_ = board.copy()
        board_[board_index] = player
        return board_

    @staticmethod
    def select_move(action_probs: dict, tau=1.0):
        """
        Selects a move from the action probabilities using either greedy or stochastic policy.
        The stochastic policy uses the tau parameter to adjust the probabilities. This is based on the
        temperature parameter in DeepMind's AlphaZero paper.

        :param action_probs: A dictionary containing the action probabilities in the form of {action_index: probability}.
        :param tau: The temperature parameter. 0 for greedy, >0 for stochastic.
        :return: The selected move as an integer (index).
        """
        if tau == 0:  # select greedy
            return max(action_probs, key=action_probs.get)  # return the key with max value
        else:  # select stochastic
            moves, probabilities = zip(*action_probs.items())
            adjusted_probs = [prob ** (1 / tau) for prob in probabilities]
            adjusted_probs_sum = sum(adjusted_probs)
            normalized_probs = [prob / adjusted_probs_sum for prob in adjusted_probs]
            return np.random.choice(moves, p=normalized_probs)

    def __str__(self):
        return str(self.board).replace('1', 'X').replace('-1', 'O')
