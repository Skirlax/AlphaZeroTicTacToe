import random
import sys

import numpy as np
import pygame as pg


class GameManager:
    """
    This class is the game manager for the game of Tic Tac Toe and its variants.
    """

    def __init__(self, board_size: int, headless: bool, num_to_win=None) -> None:
        # TODO: Implement the possibility to play over internet using sockets.
        self.player = 1
        self.enemy_player = -1
        self.board_size = board_size
        self.board = self.initialize_board()
        self.headless = headless
        self.num_to_win = self.init_num_to_win(num_to_win)
        self.screen = self.create_screen(headless)

    def play(self, player: int, index: tuple) -> None:
        self.board[index] = player

    def init_num_to_win(self, num_to_win: int | None) -> int:
        if num_to_win is None:
            num_to_win = self.board_size
        if num_to_win > self.board_size:
            raise Exception("Num to win can't be greater than board size")
        return num_to_win

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

    def full_iterate_array_all_diags(self, arr: np.ndarray, func: callable):

        results = []
        for row in arr:
            results.append(func(row.reshape(-1)))

        for col in arr.T:
            results.append(func(col.reshape(-1)))

        diags = [np.diag(arr, k=i)
                 for i in range(-arr.shape[0] + 1, arr.shape[1])]
        flipped_diags = [np.diag(np.fliplr(arr), k=i)
                         for i in range(-arr.shape[0] + 1, arr.shape[1])]
        diags.extend(flipped_diags)
        for diag in diags:
            # if diag.size < self.num_to_win:
            #     continue
            results.append(func(diag.reshape(-1)))

        return results

    def check_win(self, player: int, board=None) -> bool:
        if self.num_to_win == self.board_size:
            return self.check_full_win(player, board=board)
        else:
            return self.check_partial_win(player, self.num_to_win, board=board)

    def check_full_win(self, player: int, board=None) -> bool:
        """
        This function checks if the supplied player has won the game with a full win (num_to_win == board_size).
        :param player: The player to check for (1 or -1).
        :param board: The board to check on. If None, the current board is used.
        :return: True if the player has won, False otherwise.
        """
        if board is None:
            board = self.get_board()
        matches = self.full_iterate_array(
            board, lambda part: np.all(part == player))
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
        if board is None:
            board = self.get_board()
        matches = self.full_iterate_array_all_diags(board,
                                                    lambda part:
                                                    np.convolve((part == player), np.ones(n, dtype=np.int8),
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
            if index is str:
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

    def render(self) -> bool:
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
                pg.draw.rect(self.screen, (255, 255, 255),
                             pg.Rect(x, y, 100, 100), 1)

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

    def get_click_coords(self):
        if self.headless:
            return
        mouse_pos = (x // 100 for x in pg.mouse.get_pos())
        if pg.mouse.get_pressed()[0]:  # left click
            return mouse_pos

    def get_human_input(self, board: np.ndarray):
        if self.headless:
            return
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.pygame_quit()
                    sys.exit(0)
            if self.get_click_coords() is not None:
                x, y = self.get_click_coords()
                if board[y][x] == 0:
                    return y, x

            # time.sleep(1 / 60)

    def game_result(self, player, board) -> float | None:
        """
        Returns the result of the game from the perspective of the supplied player.
        :param player: The player to check for (1 or -1).
        :param board: The board to check on. If None, the current board is used.
        :return: The game result. None when the game is not over yet.
        """
        if self.check_win(player, board=board):
            return 1.0
        elif self.check_win(-player, board=board):
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
        else:  # select stochastic
            moves, probabilities = zip(*action_probs.items())
            adjusted_probs = [prob ** (1 / tau) for prob in probabilities]
            adjusted_probs_sum = sum(adjusted_probs)
            normalized_probs = [
                prob / adjusted_probs_sum for prob in adjusted_probs]
            return dict(zip(moves, normalized_probs))

    @staticmethod
    def select_move(action_probs: dict):
        moves, probs = zip(*action_probs.items())
        return np.random.choice(moves, p=probs)

    def __str__(self):
        return str(self.board).replace('1', 'X').replace('-1', 'O')

# if __name__ == "__main__":
#     sample_arr = np.array([[-1,1,-1,1,1],[-1,1,1,-1,-1],[1,-1,1,1,1],[-1,1,1,-1,1],[0,0,-1,1,1]])
#     game_manager = GameManager(5, True,num_to_win=3)
#     res = game_manager.game_result(-1,sample_arr)
#     print(res)
