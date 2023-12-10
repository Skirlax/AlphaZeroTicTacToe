import CpSelfPlay
from copy import deepcopy

import joblib
import numpy as np
import torch as th
from joblib import Parallel, delayed
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from AlphaZero.Arena.arena import Arena
from AlphaZero.Arena.players import NetPlayer, RandomPlayer, Player
from AlphaZero.MCTS.search_tree import McSearchTree
from AlphaZero.Network.nnet import TicTacToeNet
from AlphaZero.checkpointer import CheckPointer
from AlphaZero.logger import LoggingMessageTemplates, Logger
from AlphaZero.utils import check_args, DotDict, mask_invalid_actions_batch, build_net_from_args, build_all_from_args, \
    upload_checkpoint_to_gdrive
from Game.game import GameManager
from mem_buffer import MemBuffer

joblib.parallel.BACKENDS['multiprocessing'].use_dill = True
from multiprocessing import set_start_method
import pickle

set_start_method('spawn', force=True)


class Trainer:
    def __init__(self, network: TicTacToeNet, optimizer: th.optim,
                 memory: MemBuffer, args: DotDict,
                 checkpointer: CheckPointer, device,
                 headless: bool = True) -> None:
        check_args(args)
        self.args = args
        self.device = device
        self.headless = headless
        self.game_manager = GameManager(board_size=self.args.board_size, headless=headless,
                                        num_to_win=self.args.num_to_win)
        self.mcts = McSearchTree(self.game_manager, self.args)
        self.network = network
        self.opponent_network = build_net_from_args(args, device)
        self.optimizer = optimizer
        self.memory = memory
        self.summary_writer = SummaryWriter("Logs/AlphaZero")
        self.mse_loss = th.nn.MSELoss()
        self.arena = Arena(self.game_manager, self.args, self.device)
        self.checkpointer = checkpointer
        self.logger = Logger()
        self.arena_win_frequencies = []

    @classmethod
    def from_checkpoint(cls, args: dict, checkpoint_num: int, headless: bool = True,
                        checkpointer_verbose: bool = False):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        checkpointer = CheckPointer(args["checkpoint_dir"], verbose=checkpointer_verbose)
        if checkpoint_num < 0:
            highest_num = checkpointer.get_highest_checkpoint_num()
            if highest_num is None:
                raise FileNotFoundError("No checkpoints found, nothing to restore.")
            if checkpoint_num < -1:
                checkpoint_num = highest_num + checkpoint_num
            else:
                checkpoint_num = highest_num

        network_dict, optimizer_dict, memory, lr = checkpointer.load_checkpoint_from_num(checkpoint_num)
        network, optimizer, _ = build_all_from_args(args, device, lr=lr)
        del _
        network.load_state_dict(network_dict)
        optimizer.load_state_dict(optimizer_dict)
        return cls(network, optimizer, memory, args, checkpointer, device, headless=headless)

    @classmethod
    def from_latest(cls, path: str, headless: bool = True, checkpointer_verbose: bool = False):
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
        return cls(network, optimizer, memory, args, checkpointer, device, headless=headless)

    @classmethod
    def create(cls, args: dict, headless: bool = True, checkpointer_verbose: bool = False):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        net, optimizer, memory = build_all_from_args(args, device)
        checkpointer = CheckPointer(args["checkpoint_dir"], verbose=checkpointer_verbose)
        return cls(net, optimizer, memory, args, checkpointer, device, headless=headless)

    @classmethod
    def from_state_dict(cls, path: str, args: dict, headless: bool = True, checkpointer_verbose: bool = False):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        net, optimizer, memory = build_all_from_args(args, device)
        checkpointer = CheckPointer(args["checkpoint_dir"], verbose=checkpointer_verbose)
        net.load_state_dict(th.load(path))
        return cls(net, optimizer, memory, args, checkpointer, device, headless=headless)

    def pi_loss(self, y_hat, y, masks):
        masks = masks.reshape(y_hat.shape).to(self.device)
        masked_y_hat = masks * y_hat
        return -th.sum(y * masked_y_hat) / y.size()[0]

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
            with th.no_grad():
                self.logger.log(LoggingMessageTemplates.SELF_PLAY_START(self_play_games))
                args_ = self.args.copy()
                args_.pop("max_depth")
                args_.pop("checkpoint_dir")
                traced = self.network.to_traced_script()
                traced_save_path = "/home/skyr/PycharmProjects/AlphaZeroTicTacToe/traced.pt"
                traced.save(traced_save_path)
                history, wins_p1, wins_p2, game_draws = CpSelfPlay.CparallelSelfPlay(self.args["num_workers"],
                                                                                     self_play_games, traced_save_path,
                                                                                     args_)
                with open("history.pkl", "wb") as f:
                    pickle.dump(history, f)
                # wins_p1, wins_p2, game_draws = self.parallel_self_play(self.args.num_workers, self_play_games)
                # wins_p1, wins_p2, game_draws = 0, 0, 0
                # for j in self.make_tqdm_bar(range(self_play_games), "Self-Play Progress", 1, leave=False):
                #     game_history, wins_one, wins_minus_one, draws = self.mcts.play_one_game(self.network, self.device)
                #     # print(f"Game {j + 1} finished.")
                #     self.mcts.step_root(None)  # reset the search tree
                #     self.memory.add_list(game_history)
                #     wins_p1 += wins_one
                #     wins_p2 += wins_minus_one
                #     game_draws += draws
                self.logger.log(LoggingMessageTemplates.SELF_PLAY_END(wins_p1, wins_p2, game_draws))

            self.summary_writer.add_scalar("Self-Play Win Percentage Player One", wins_p1 / self_play_games, i)
            self.summary_writer.add_scalar("Self-Play Loss Percentage Player One",
                                           wins_p2 / self_play_games, i)
            self.summary_writer.add_scalar("Self-Play Draw Percentage", game_draws / self_play_games, i)

            self.checkpointer.save_temp_net_checkpoint(self.network)
            self.checkpointer.load_temp_net_checkpoint(self.opponent_network)
            self.logger.log(LoggingMessageTemplates.SAVED("temp checkpoint", self.checkpointer.get_temp_path()))

            self.network.train()
            self.memory.shuffle()
            self.logger.log(LoggingMessageTemplates.NETWORK_TRAINING_START(epochs))
            mean_loss = self.train_network(epochs, i, batch_size)
            self.checkpointer.save_checkpoint(self.network, self.optimizer, self.memory,
                                              self.args["lr"], i, self.args, name="latest_trained_net")
            # upload_checkpoint_to_gdrive([self.checkpointer.get_latest_name_match("latest_trained_net"),
            #                              self.checkpointer.get_latest_name_match(self.checkpointer.get_name_prefix())],
            #                             not_notebook_ok=True)

            self.logger.log(LoggingMessageTemplates.NETWORK_TRAINING_END(mean_loss))

            self.logger.log(LoggingMessageTemplates.LOADED("opponent network", self.checkpointer.get_temp_path()))
            self.network.eval()
            self.opponent_network.eval()
            p1_game_manager = GameManager(self.args["board_size"], self.headless, num_to_win=self.args.num_to_win)
            p2_game_manager = GameManager(self.args["board_size"], self.headless, num_to_win=self.args.num_to_win)
            p1_tree = McSearchTree(p1_game_manager, self.args)
            p2_tree = McSearchTree(p2_game_manager, self.args)
            p1 = NetPlayer(self.network, p1_tree, p1_game_manager)
            p2 = NetPlayer(self.opponent_network, p2_tree, p2_game_manager)
            num_games = self.args["num_pit_games"]
            update_threshold = self.args["update_threshold"]
            self.logger.log(LoggingMessageTemplates.PITTING_START(p1.name, p2.name, num_games))
            p1_wins, p2_wins, draws = self.arena.pit(p1, p2, num_games, num_mc_simulations=num_simulations)
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
                    random_player = RandomPlayer(self.game_manager)
                    p1 = NetPlayer(self.network, self.mcts, self.game_manager)
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
                self.checkpointer.save_checkpoint(self.network, self.optimizer, self.memory, self.args["lr"], i,
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

    def only_pit(self, p1: Player, p2: Player, num_games: int):
        if p1 == NetPlayer and p2 == NetPlayer:
            p1_manager = GameManager(self.args["board_size"], self.headless, num_to_win=self.args.num_to_win)
            p1_tree = McSearchTree(p1_manager, self.args)
            p2_manager = GameManager(self.args["board_size"], self.headless, num_to_win=self.args.num_to_win)
            p2_tree = McSearchTree(p2_manager, self.args)
            p1 = NetPlayer(self.network, p1_tree, p1_manager)
            p2 = NetPlayer(self.opponent_network, p2_tree, p2_manager)

        # TODO: Handle other cases.

        num_simulations = self.args["num_simulations"]
        self.arena.pit(p1, p2, num_games, num_mc_simulations=num_simulations)

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
        n_games_per_job = n_games // n_jobs
        results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(self_play)(networks[i], n_games_per_job, trees[i], self.device)
            for i in range(n_jobs))
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
            manager = GameManager(self.args["board_size"], self.headless, num_to_win=self.args.num_to_win)
            trees.append(McSearchTree(manager, dict(self.args)))
        return trees

    def train_network(self, epochs: int, i: int, batch_size: int) -> float:
        """
        Trains self.network for the given number of epochs.

        :param epochs: The number of epochs to train for.
        :param i: Current training iteration.
        :param batch_size: The batch size to use.
        :return: The mean loss over all epochs.
        """
        losses = []
        self.optimizer = th.optim.Adam(self.network.parameters(), lr=self.args["lr"])
        for epoch in self.make_tqdm_bar(range(epochs), "Network Training Progress", 1, leave=False):
            for experience_batch in self.memory(batch_size):
                if len(experience_batch) <= 1:
                    continue
                states, pi, v = zip(*experience_batch)
                states = th.tensor(np.array(states), dtype=th.float32, device=self.device)
                pi = th.tensor(np.array(pi), dtype=th.float32, device=self.device)
                v = th.tensor(v, dtype=th.float32, device=self.device).unsqueeze(1)
                pi_pred, v_pred = self.network(states)
                masks = mask_invalid_actions_batch(states)
                loss = self.mse_loss(v_pred, v) + self.pi_loss(pi_pred, pi, masks)
                losses.append(loss.item())
                self.summary_writer.add_scalar("Loss", loss.item(), i * epochs + epoch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return sum(losses) / len(losses)

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
            'optimizer_state_dict': optimizer_state_dict,
            'memory': self.memory,
            'lr': self.args["lr"],
            'state_dict': state_dict,
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


def self_play(network: TicTacToeNet, num_games: int, tree: McSearchTree, device):
    results = []
    for game in range(num_games):
        game_history, wins_one, wins_two, draws = tree.play_one_game(network, device)
        tree.step_root(None)
        results.append([game_history, wins_one, wins_two, draws])
    return results
