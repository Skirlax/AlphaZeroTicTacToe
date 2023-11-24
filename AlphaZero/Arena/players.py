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

        pi, _ = self.monte_carlo_tree_search.search(self.network, board, current_player, device, tau=tau)
        move = self.game_manager.select_move(pi)
        # visualize_tree(self.monte_carlo_tree_search.root_node, output_file_name="tree_viz",depth_limit=2)
        self.monte_carlo_tree_search.step_root(None)
        return self.game_manager.network_to_board(move)


class HumanPlayer(Player):
    def __init__(self, game_manager: GameManager):
        self.name = "HumanPlayer"
        self.game_manager = game_manager

    def choose_move(self, board: np.ndarray, **kwargs) -> tuple[int, int]:
        if self.game_manager.headless:
            raise RuntimeError("Cannot play with a human player in headless mode.")
        move = self.game_manager.get_human_input(board)
        return move


class MinimaxPlayer(Player):
    def __init__(self, game_manager: GameManager, evaluate_fn: callable):
        self.evaluate_fn = evaluate_fn
        self.game_manager = game_manager
        self.name = "MinimaxPlayer"

    def choose_move(self, board: np.ndarray, **kwargs) -> tuple[int, int]:
        try:
            depth = kwargs["depth"]
            player = kwargs["player"]
        except KeyError:
            raise KeyError("Missing keyword argument. Please supply kwargs: depth, player")
        move = self.minimax(board, depth, True, player)[1]
        return tuple(move)

    def minimax(self, board: np.ndarray, depth: int, is_max: bool, player: int, alpha=-float("inf"),
                beta=float("inf")) -> tuple:
        if depth == 0:
            return self.evaluate_fn(board, -1), None

        if is_max:
            best_score = -float("inf")
            best_move = None
            for move in self.game_manager.get_legal_moves_on_observation(board):
                board[move[0]][move[1]] = player
                score = self.minimax(board.copy(), depth - 1, False, -player, alpha, beta)[0]
                if score > best_score:
                    best_move = move
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)

                if alpha >= beta:
                    break
                board[move[0]][move[1]] = 0
            return best_score if best_score != -float("inf") else self.evaluate_fn(board, player), best_move
        else:
            best_score = float("inf")
            best_move = None
            for move in self.game_manager.get_legal_moves_on_observation(board):
                board[move[0]][move[1]] = player
                score = self.minimax(board.copy(), depth - 1, True, -player, alpha, beta)[0]
                if score < best_score:
                    best_move = move
                best_score = min(score, best_score)
                beta = min(beta, best_score)

                if beta <= alpha:
                    break
                board[move[0]][move[1]] = 0
            return best_score if best_score != float("inf") else self.evaluate_fn(board, player), best_move
