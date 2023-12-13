import datetime

import torch as th

from AlphaZero.MCTS.node import Node
from AlphaZero.utils import augment_experience_with_symmetries, mask_invalid_actions, DotDict
from Game.game import GameManager


class McSearchTree:
    def __init__(self, game_manager: GameManager, args: DotDict):
        self.game_manager = game_manager
        self.args = args
        self.root_node = None

    def play_one_game(self, network, device) -> tuple[list, int, int, int]:
        """
        Plays a single game using the Monte Carlo Tree Search algorithm.

        Args:
            network: The neural network used for searching and evaluating moves.
            device: The device (e.g., CPU or GPU) on which the network is located.

        Returns:
            A tuple containing the game history, and the number of wins, losses, and draws.
            The game history is a list of tuples, where each tuple contains:
            - The game state multiplied by the current player
            - The policy vector (probability distribution over moves)
            - The game result (1 for a win, -1 for a loss, 0 for a draw)
            - The current player (-1 or 1)
        """
        # tau = self.args["tau"]
        state = self.game_manager.reset()
        current_player = 1
        game_history = []
        results = {"1": 0, "-1": 0, "D": 0}
        while True:
            pi, _ = self.search(network, state, current_player, device)
            move = self.game_manager.select_move(pi)
            # self.step_root([move])
            self.step_root(None)
            self.game_manager.play(current_player, self.game_manager.network_to_board(move))
            # pi = [x for x in pi.values()]
            game_history.append((state * current_player, pi, None, current_player))
            state = self.game_manager.get_board()
            r = self.game_manager.game_result(current_player, state)
            if r is not None:
                if r == current_player:
                    results["1"] += 1
                elif -1 < r < 1:
                    results["D"] += 1
                else:
                    results["-1"] += 1

                if -1 < r < 1:
                    game_history = [(x[0], x[1], r, x[3]) for x in game_history]
                else:
                    game_history = [(x[0], x[1], r * current_player * x[3], x[3]) for x in game_history]
                break
            current_player *= -1

        # game_history = make_channels(game_history)
        game_history = augment_experience_with_symmetries(game_history, self.game_manager.board_size)
        return game_history, results["1"], results["-1"], results["D"]

    def search(self, network, state, current_player, device, tau=None):
        """
        Perform a Monte Carlo Tree Search on the current state starting with the current player.
        :param tau:
        :param network:
        :param state:
        :param current_player:
        :param device:
        :return:
        """
        num_simulations = self.args["num_simulations"]
        if tau is None:
            tau = self.args["tau"]
        if self.root_node is None:
            self.root_node = Node(current_player, times_visited_init=0)
        state_ = self.game_manager.get_canonical_form(state, current_player)
        # state_ = make_channels_from_single(state_)
        state_ = th.tensor(state_, dtype=th.float32, device=device).unsqueeze(0)
        probabilities, v = network.predict(state_)

        probabilities = mask_invalid_actions(probabilities, state, self.game_manager.board_size)
        probabilities = probabilities.flatten().tolist()
        self.root_node.expand(state, probabilities)
        for simulation in range(num_simulations):
            current_node = self.root_node
            path = [current_node]
            action = None
            while current_node.was_visited():
                current_node, action = current_node.get_best_child(c=self.args["c"])
                if current_node is None:  # This was for testing purposes
                    th.save(self.root_node, "root_node.pt")
                    th.save(network.state_dict(), f"network_none_checkpoint_{current_player}.pt")
                    raise ValueError("current_node is None")
                path.append(current_node)

            # leaf node reached
            next_state = self.game_manager.get_next_state(current_node.parent.state,
                                                          self.game_manager.network_to_board(action),
                                                          current_node.parent.current_player)
            next_state_ = self.game_manager.get_canonical_form(next_state, current_node.current_player)
            v = self.game_manager.game_result(current_node.current_player, next_state)
            # if v is None and (
            #         self.game_manager.check_partial_win(1, self.args["num_to_win"],
            #                                             board=next_state) or self.game_manager.check_partial_win(
            #     -1, self.args["num_to_win"], board=next_state)):
            #     print(f"Partial win found, but not detected by game_result."
            #           f"At [{datetime.datetime.now()}]. Adding possibly important info:\n"
            #           f"Current node player: {current_node.current_player}\n"
            #           f"Next state: {next_state}\n",
            #
            #           file=open("win_info.txt", "a+"))
            #     raise ValueError("Partial win found, but not detected by game_result.")
            # None
            if v is None:
                # next_state_ = make_channels_from_single(next_state_)
                next_state_ = th.tensor(next_state_, dtype=th.float32, device=device).unsqueeze(0)
                probabilities, v = network.predict(next_state_)
                probabilities = mask_invalid_actions(probabilities, next_state, self.game_manager.board_size)
                v = v.flatten().tolist()[0]
                probabilities = probabilities.flatten().tolist()
                current_node.expand(next_state, probabilities)

            self.backprop(v, path)

        return self.root_node.get_self_action_probabilities(tau=tau), self.root_node.get_self_value()

    def backprop(self, v, path):
        """
        Backpropagates the value `v` through the search tree, updating the relevant nodes.

        Args:
            v (float): The value to be backpropagated.
            path (list): The path from the leaf node to the root node.

        Returns:
            None
        """
        for node in reversed(path):
            v *= -1
            node.total_value += v
            node.update_q(v)
            node.times_visited += 1

    def step_root(self, actions: list | None) -> None:
        if actions is not None:
            if self.root_node is not None:
                if not self.root_node.was_visited():
                    return
                for action in actions:
                    self.root_node = self.root_node.children[action]
                self.root_node.parent = None
        else:
            # reset root node
            self.root_node = None
