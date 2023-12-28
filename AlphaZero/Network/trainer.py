# import CpSelfPlay
from copy import deepcopy
from multiprocessing import Pool
from typing import Type

import joblib
import torch as th
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from AlphaZero.Arena.arena import Arena
from AlphaZero.Arena.players import RandomPlayer, Player
from AlphaZero.MCTS.az_search_tree import McSearchTree
from AlphaZero.Network.nnet import TicTacToeNet
from AlphaZero.checkpointer import CheckPointer
from AlphaZero.logger import LoggingMessageTemplates, Logger
from AlphaZero.utils import check_args, DotDict, build_net_from_args, build_all_from_args
from General.az_game import Game
from General.network import GeneralNetwork
from General.search_tree import SearchTree
from mem_buffer import MemBuffer

joblib.parallel.BACKENDS['multiprocessing'].use_dill = True
from multiprocessing import set_start_method

set_start_method('spawn', force=True)


class Trainer:
    def __init__(self, network: Type[GeneralNetwork], game: Game,
                 optimizer: th.optim, memory: MemBuffer,
                 args: DotDict, checkpointer: CheckPointer,
                 search_tree: SearchTree, net_player: Player,
                 device, headless: bool = True,
                 opponent_network_override: th.nn.Module or None = None) -> None:
        check_args(args)
        self.args = args
        self.device = device
        self.headless = headless
        self.game_manager = game
        self.mcts = search_tree
        self.net_player = net_player
        self.network = network
        self.opponent_network = self.network.make_fresh_instance() if opponent_network_override is None else opponent_network_override
        self.optimizer = optimizer
        self.memory = memory
        self.summary_writer = SummaryWriter("Logs/AlphaZero")
        self.arena = Arena(self.game_manager, self.args, self.device)
        self.checkpointer = checkpointer
        self.logger = Logger()
        self.arena_win_frequencies = []

    @classmethod
    def from_checkpoint(cls, net_class: GeneralNetwork, tree_class: Type[McSearchTree], net_player_class: Type[Player],
                        checkpoint_path: str, checkpoint_dir: str,
                        game: Game, headless: bool = True,
                        checkpointer_verbose: bool = False):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        checkpointer = CheckPointer(checkpoint_dir, verbose=checkpointer_verbose)

        network_dict, optimizer_dict, memory, lr, args, opponent_dict = checkpointer.load_checkpoint_from_path(
            checkpoint_path)
        tree = tree_class(game.make_fresh_instance(), args)
        network = net_class.make_from_args(args)
        opponent_network = network.make_fresh_instance()
        optimizer = th.optim.Adam(network.parameters(), lr=lr)
        # opponent_network = build_net_from_args(args, device)
        net_player = net_player_class(game.make_fresh_instance(),
                                      **{"network": network, "monte_carlo_tree_search": tree})
        network.load_state_dict(network_dict)
        opponent_network.load_state_dict(opponent_dict)
        optimizer.load_state_dict(optimizer_dict)
        return cls(network, game, optimizer, memory, args, checkpointer, tree, net_player, device, headless=headless)

    @classmethod
    def from_latest(cls, path: str, game: Game, headless: bool = True, checkpointer_verbose: bool = False):
        data = th.load(path)

        args = data.pop("args")
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        network, optimizer, _ = build_all_from_args(args, device, lr=data["lr"])
        del _
        opponent_network = build_net_from_args(args, device)
        opponent_network.load_state_dict(data.pop("opponent_state_dict"))
        network.load_state_dict(data.pop('state_dict'))
        optimizer.load_state_dict(data.pop('optimizer_state_dict'))

        memory = data.pop("memory")
        args["lr"] = data.pop("lr")
        checkpointer = CheckPointer(args["checkpoint_dir"], verbose=checkpointer_verbose)
        return cls(network, game, optimizer, memory, args, checkpointer, device, headless=headless)

    @classmethod
    def create(cls, args: dict, game: Game, network: Type[GeneralNetwork],search_tree: SearchTree, net_player: Player, headless: bool = True,
               checkpointer_verbose: bool = False):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        _, optimizer, memory = build_all_from_args(args, device)
        checkpointer = CheckPointer(args["checkpoint_dir"], verbose=checkpointer_verbose)
        args = DotDict(args)
        return cls(network, game, optimizer, memory, args, checkpointer, search_tree, net_player, device, headless=headless)

    @classmethod
    def from_state_dict(cls, path: str, args: dict, game: Game, headless: bool = True,
                        checkpointer_verbose: bool = False):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        net, optimizer, memory = build_all_from_args(args, device)
        checkpointer = CheckPointer(args["checkpoint_dir"], verbose=checkpointer_verbose)
        net.load_state_dict(th.load(path))
        return cls(net, game, optimizer, memory, args, checkpointer, device, headless=headless)

    def train(self) -> TicTacToeNet:
        self.logger.log(LoggingMessageTemplates.TRAINING_START(self.args["num_iters"]))
        self.opponent_network.load_state_dict(self.network.state_dict())
        # self.checkpointer.save_state_dict_checkpoint(self.network, "h_search_network")
        num_iters = self.args["num_iters"]
        epochs = self.args["epochs"]
        batch_size = self.args["batch_size"]
        num_simulations = self.args["num_simulations"]
        self_play_games = self.args["self_play_games"]
        self.network.eval()

        for i in self.make_tqdm_bar(range(num_iters), "Training Progress", 0):
            if i >= self.args["zero_tau_after"]:
                self.args["arena_tau"] = 0
            with th.no_grad():
                self.logger.log(LoggingMessageTemplates.SELF_PLAY_START(self_play_games))
                # wins_p1, wins_p2, game_draws = self.parallel_self_play(self.args["num_workers"], self_play_games)
                history,wins_p1, wins_p2, game_draws = self_play(
                    (self.network, self.mcts, self.args["self_play_games"], self.device))
                self.logger.log(LoggingMessageTemplates.SELF_PLAY_END(wins_p1, wins_p2, game_draws))
                self.memory.add_list(history)

            self.summary_writer.add_scalar("Self-Play Win Percentage Player One", wins_p1 / self_play_games, i)
            self.summary_writer.add_scalar("Self-Play Loss Percentage Player One",
                                           wins_p2 / self_play_games, i)
            self.summary_writer.add_scalar("Self-Play Draw Percentage", game_draws / self_play_games, i)

            self.checkpointer.save_temp_net_checkpoint(self.network)
            self.checkpointer.load_temp_net_checkpoint(self.opponent_network)
            self.logger.log(LoggingMessageTemplates.SAVED("temp checkpoint", self.checkpointer.get_temp_path()))

            self.network.train()
            # self.memory.shuffle()
            self.logger.log(LoggingMessageTemplates.NETWORK_TRAINING_START(epochs))
            mean_loss = self.network.train_net(self.memory, self.args)
            self.checkpointer.save_checkpoint(self.network, self.opponent_network, self.optimizer, self.memory,
                                              self.args["lr"], i, self.args, name="latest_trained_net")

            self.logger.log(LoggingMessageTemplates.NETWORK_TRAINING_END(mean_loss))

            self.logger.log(LoggingMessageTemplates.LOADED("opponent network", self.checkpointer.get_temp_path()))
            self.network.eval()
            self.opponent_network.eval()
            # p1_game_manager = self.game_manager.make_fresh_instance()
            # p2_game_manager = self.game_manager.make_fresh_instance()
            # p1_tree = self.mcts.make_fresh_instance()
            # p2_tree = self.mcts.make_fresh_instance()
            # p1 = NetPlayer(self.network, p1_tree, p1_game_manager)
            p1 = self.net_player.make_fresh_instance()
            p1.set_network(self.network)
            # p2 = NetPlayer(self.opponent_network, p2_tree, p2_game_manager)
            p2 = self.net_player.make_fresh_instance()  # **{"network": self.opponent_network, "monte_carlo_tree_search": p2_tree}
            p2.set_network(self.opponent_network)
            num_games = self.args["num_pit_games"]
            update_threshold = self.args["update_threshold"]
            self.logger.log(LoggingMessageTemplates.PITTING_START(p1.name, p2.name, num_games))
            p1_wins, p2_wins, draws = self.arena.pit(p1, p2, num_games, num_mc_simulations=num_simulations,
                                                     one_player=False)
            self.logger.log(LoggingMessageTemplates.PITTING_END(p1.name, p2.name, p1_wins,
                                                                p2_wins, draws))
            self.arena_win_frequencies.append(p1_wins / num_games)
            wins_total = self.not_zero(p1_wins + p2_wins)
            self.summary_writer.add_scalar("Net_vs_Net Win Percentage Player One", p1_wins / wins_total,
                                           i)
            self.summary_writer.add_scalar("Net_vs_Net Loss Percentage Player One",
                                           p2_wins / wins_total, i)
            self.summary_writer.add_scalar("Net_vs_Net Draw Percentage", draws / num_games, i)

            if i % self.args["random_pit_freq"] == 0:
                self.network.eval()
                with th.no_grad():
                    random_player = RandomPlayer(self.game_manager.make_fresh_instance(), **{})
                    # self.network, self.mcts,
                    p1 = self.net_player.make_fresh_instance()  # **{"network": self.network, "monte_carlo_tree_search": self.mcts}
                    p1.set_network(self.network)
                    self.logger.log(LoggingMessageTemplates.PITTING_START(p1.name, random_player.name, num_games))
                    p1_wins_random, p2_wins_random, draws_random = self.arena.pit(p1, random_player, num_games,
                                                                                  num_mc_simulations=num_simulations)
                    self.logger.log(
                        LoggingMessageTemplates.PITTING_END(p1.name, random_player.name, p1_wins_random,
                                                            p2_wins_random, draws_random))
                    self.summary_writer.add_scalar("Net_vs_Random Win Percentage Player One",
                                                   p1_wins_random / num_games, i)
                    self.summary_writer.add_scalar("Net_vs_Random Loss Percentage Player One",
                                                   p2_wins_random / num_games, i)
                    self.summary_writer.add_scalar("Net_vs_Random Draw Percentage",
                                                   draws_random / num_games, i)

            if p1_wins / wins_total > update_threshold:
                self.logger.log(LoggingMessageTemplates.MODEL_ACCEPT(p1_wins / wins_total,
                                                                     update_threshold))
                self.checkpointer.save_checkpoint(self.network, self.opponent_network, self.optimizer, self.memory,
                                                  self.args["lr"], i,
                                                  self.args)
                self.logger.log(LoggingMessageTemplates.SAVED("accepted model checkpoint",
                                                              self.checkpointer.get_checkpoint_dir()))
            else:
                self.logger.log(LoggingMessageTemplates.MODEL_REJECT(p1_wins / wins_total,
                                                                     update_threshold))
                self.checkpointer.load_temp_net_checkpoint(self.network)
                self.logger.log(LoggingMessageTemplates.LOADED("previous version checkpoint",
                                                               self.checkpointer.get_temp_path()))

        important_args = {
            "numIters": self.args["num_iters"],
            "numSelfPlayGames": self.args["self_play_games"],
            "temp": self.args["tau"],
            "updateThreshold": self.args["update_threshold"],
            "mcSimulations": self.args["num_simulations"],
            "c": self.args["c"],
            "maxDepth": self.args["max_depth"],
            "numPitGames": self.args["num_pit_games"]
        }

        self.logger.log(LoggingMessageTemplates.TRAINING_END(important_args))
        return self.network

    # def only_pit(self, p1: Player, p2: Player, num_games: int):
    #     if p1 == NetPlayer and p2 == NetPlayer:
    #         p1_manager = self.game_manager.make_fresh_instance()
    #         p1_tree = McSearchTree(p1_manager, self.args)
    #         p2_manager = self.game_manager.make_fresh_instance()
    #         p2_tree = McSearchTree(p2_manager, self.args)
    #         # p1 = NetPlayer(self.network, p1_tree, p1_manager)
    #         p1 = NetPlayer(p1_manager, **{"network": self.network, "monte_carlo_tree_search": p1_tree})
    #         # p2 = NetPlayer(self.opponent_network, p2_tree, p2_manager)
    #         p2 = NetPlayer(p2_manager, **{"network": self.opponent_network, "monte_carlo_tree_search": p2_tree})
    #
    #
    #     num_simulations = self.args["num_simulations"]
    #     self.arena.pit(p1, p2, num_games, num_mc_simulations=num_simulations)

    def parallel_self_play(self, n_jobs: int, n_games: int) -> tuple:
        """
        This method performs self-play using n_jobs processes. Each process has its own copy of the network and
        the search tree. The results of the games are then aggregated and returned.

        :param n_jobs: The number of processes to divide the self play games into.
        :param n_games: The total number of games to play.
        :return: Wins statistics.
        """
        networks = self.make_n_networks(n_jobs)
        trees = self.make_n_trees(n_jobs)
        num_games = n_games // n_jobs
        print(f"Starting parallel self-play with {n_jobs} processes (games per process: {num_games})")
        chunks = [(net, tree, num_games, self.device) for net, tree in zip(networks, trees)]
        with Pool(processes=n_jobs) as pool:
            results = pool.map(self_play, chunks)
        wins_one = 0
        wins_two = 0
        draws = 0
        for result in results:
            for game_history, wins_one_, wins_two_, draws_ in result:
                self.memory.add_list(game_history)
                wins_one += wins_one_
                wins_two += wins_two_
                draws += draws_
        return wins_one, wins_two, draws

    def make_n_networks(self, n: int) -> list[TicTacToeNet]:
        """
        Make n identical copies of self.network using deepcopy.

        :param n: The number of copies to make.
        :return: A list of n identical networks.
        """
        return [deepcopy(self.network) for _ in range(n)]

    def make_n_trees(self, n: int) -> list[McSearchTree]:
        """
        Make n new search trees.
        :param n: The number of trees to create.
        :return: A list of n new search trees.
        """
        trees = []
        for i in range(n):
            # manager = self.game_manager.make_fresh_instance()
            tree = self.mcts.make_fresh_instance()
            trees.append(tree)
        return trees

    def get_arena_win_frequencies_mean(self):
        return sum(self.arena_win_frequencies) / self.not_zero(len(self.arena_win_frequencies))

    def get_memory(self):
        return self.memory

    def not_zero(self, x):
        return x if x != 0 else 1

    def save_latest(self, path):
        state_dict = self.network.state_dict()
        opponent_state_dict = self.opponent_network.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        th.save({
            'optimizer': optimizer_state_dict,
            'memory': self.memory,
            'lr': self.args["lr"],
            'net': state_dict,
            'opponent_state_dict': opponent_state_dict,
            'args': self.args
        }, path)
        print("Saved latest model data to {}".format(path))

    def make_tqdm_bar(self, iterable, desc, position, leave=True):
        if self.args.show_tqdm:
            return tqdm(iterable, desc=desc, position=position, leave=leave)
        else:
            return iterable

    def get_network_mem_size(self):
        # This is approximate and doesn't consider peak usage.
        mem_params = sum([param.nelement() * param.element_size() for param in self.network.parameters()])
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.network.buffers()])
        return (mem_params + mem_bufs) / (1024 ** 2)

    def get_network(self):
        return self.network


def self_play(chunk) -> list:
    network, tree, num_games, device = chunk
    results = []
    for game in range(num_games):
        game_history, wins_one, wins_two, draws = tree.play_one_game(network, device)
        return game_history, wins_one, wins_two, draws
        tree.step_root(None)
        results.append([game_history, wins_one, wins_two, draws])
    return results
