from abc import ABC, abstractmethod

# import CpSelfPlay
import numpy as np

from AlphaZero.Network.nnet import TicTacToeNet
from Game.tictactoe_game import TicTacToeGameManager
from General.az_game import Game


class Player(ABC):
    """
    To create a custom player, extend this class and implement the choose_move method.
    You can see different implementations below.
    """

    @abstractmethod
    def __init__(self, game_manager: Game, **kwargs):
        pass

    @abstractmethod
    def choose_move(self, board: np.ndarray, **kwargs) -> tuple[int, int]:
        pass

    @abstractmethod
    def make_fresh_instance(self):
        pass

    def init_kwargs(self, kwargs: dict):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])


class RandomPlayer(Player):
    def __init__(self, game_manager: TicTacToeGameManager, **kwargs):
        self.game_manager = game_manager
        self.name = self.__class__.__name__
        self.kwargs = kwargs
        self.init_kwargs(kwargs)

    def choose_move(self, board: np.ndarray, **kwargs) -> tuple[int, int]:
        move = self.game_manager.get_random_valid_action(board)
        return tuple(move)

    def make_fresh_instance(self):
        return RandomPlayer(self.game_manager.make_fresh_instance(), **self.kwargs)


class NetPlayer(Player):
    def __init__(self, game_manager: TicTacToeGameManager, **kwargs):
        self.game_manager = game_manager
        # self.network = network
        # self.monte_carlo_tree_search = mc_tree_search
        self.name = self.__class__.__name__
        self.kwargs = kwargs
        self.init_kwargs(kwargs)

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
        # print(self.game_manager.network_to_board(move))
        # visualize_tree(self.monte_carlo_tree_search.root_node, output_file_name="tree_viz",depth_limit=2)
        self.monte_carlo_tree_search.step_root(None)
        return self.game_manager.network_to_board(move)

    def make_fresh_instance(self):
        return NetPlayer(self.game_manager.make_fresh_instance(), **{"network": self.network,
                                                                     "monte_carlo_tree_search": self.monte_carlo_tree_search.make_fresh_instance()})

    def set_network(self, network):
        self.network = network


class TrainingNetPlayer(Player):
    def __init__(self, network: TicTacToeNet, game_manager: TicTacToeGameManager, args: dict):
        raise NotImplementedError("Don't use this class yet, it produces incorrect results.")
        self.name = self.__class__.__name__
        self.args = self.__init_args(args)
        self.network = network
        self.game_manager = game_manager
        self.traced_path = self.network.trace(self.args["board_size"])

    def __init_args(self, args) -> dict:
        for key in ["checkpoint_dir", "max_depth"]:
            try:
                args.pop(key)
            except KeyError:
                print(f"Key {key} not present.")
        return args

    def choose_move(self, board: np.ndarray, **kwargs) -> tuple[int, int]:
        try:
            current_player = kwargs["current_player"]
            device = kwargs["device"]
            tau = kwargs["tau"]
        except KeyError:
            raise KeyError("Missing keyword argument. Please supply kwargs: current_player, device, "
                           "tau")
        pi = CpSelfPlay.CmctsSearch(board, current_player, tau, self.args, self.traced_path)
        move = self.game_manager.select_move(pi)
        return self.game_manager.network_to_board(move)

    def make_fresh_instance(self):
        raise NotImplementedError


class HumanPlayer(Player):
    def __init__(self, game_manager: TicTacToeGameManager, **kwargs):
        self.name = self.__class__.__name__
        self.game_manager = game_manager
        self.kwargs = kwargs
        self.init_kwargs(kwargs)

    def choose_move(self, board: np.ndarray, **kwargs) -> tuple[int, int]:
        if self.game_manager.headless:
            raise RuntimeError("Cannot play with a human player in headless mode.")
        move = self.game_manager.get_human_input(board)
        return move

    def make_fresh_instance(self):
        return HumanPlayer(self.game_manager.make_fresh_instance(), **self.kwargs)


class MinimaxPlayer(Player):
    def __init__(self, game_manager: TicTacToeGameManager, **kwargs):
        # self.evaluate_fn = evaluate_fn
        self.game_manager = game_manager
        self.name = self.__class__.__name__
        self.kwargs = kwargs
        self.init_kwargs(kwargs)

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
        self.game_manager.check_pg_events()
        if depth == 0:
            return self.evaluate_fn(board, player), None

        if is_max:
            best_score = -float("inf")
            best_move = None
            for move in self.game_manager.get_valid_moves(board):
                board[move[0]][move[1]] = player
                score = self.minimax(board.copy(), depth - 1, False, -player, alpha, beta)[0]
                if score > best_score:
                    best_move = move
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)

                if alpha >= beta:
                    break
                board[move[0]][move[1]] = 0
            return best_score, best_move
        else:
            best_score = float("inf")
            best_move = None
            for move in self.game_manager.get_valid_moves(board):
                board[move[0]][move[1]] = player
                score = self.minimax(board.copy(), depth - 1, True, -player, alpha, beta)[0]
                if score < best_score:
                    best_move = move
                best_score = min(score, best_score)
                beta = min(beta, best_score)

                if beta <= alpha:
                    break
                board[move[0]][move[1]] = 0
            return best_score, best_move

    def make_fresh_instance(self):
        return MinimaxPlayer(self.game_manager.make_fresh_instance(), **self.kwargs)
