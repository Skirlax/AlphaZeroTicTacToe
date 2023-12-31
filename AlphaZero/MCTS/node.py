import math

import numpy as np

from Game.tictactoe_game import TicTacToeGameManager as GameManager


class Node:
    """
    This class defines a node in the search tree. It stores all the information needed for DeepMind's AlphaZero algorithm.
    """

    def __init__(self, current_player, select_probability=0, parent=None, times_visited_init=0):
        self.times_visited = times_visited_init
        self.was_init_with_zero_visits = times_visited_init == 0
        self.children = {}
        self.parent = parent
        self.select_probability = select_probability
        self.q = None
        self.current_player = current_player
        self.state = None
        self.total_value = 0

    def expand(self, state, action_probabilities) -> None:
        """
        Expands the newly visited node with the given action probabilities and state.
        :param state: np.ndarray of shape (board_size, board_size) representing the state current game board.
        :param action_probabilities: list of action probabilities for each action.
        :return: None
        """

        self.state = state.copy()  # copying here is probably not necessary.

        for action, probability in enumerate(action_probabilities):
            node = Node(self.current_player * (-1), select_probability=probability, parent=self)
            self.children[action] = node

    def was_visited(self):
        return len(self.children) > 0

    def update_q(self, v):
        # Based on DeepMind's AlphaZero paper.
        if self.q is None:
            self.q = v
        else:
            self.q = (self.times_visited * self.q + v) / (self.times_visited + 1)

    def get_best_child(self, c=1.5):
        best_utc = -float("inf")
        best_child = None
        best_action = None
        valids_for_state = np.where(self.state != 0, -5, self.state)
        valids_for_state = np.where(valids_for_state == 0, 1, valids_for_state)
        utcs = []
        for action, child in self.children.items():
            action_ = np.unravel_index(action, self.state.shape)
            if valids_for_state[action_] != 1:
                continue
            child_utc = child.calculate_utc(c=c)
            utcs.append(child_utc)
            if child_utc > best_utc:
                best_utc = child_utc
                best_child = child
                best_action = action

        printable_children = [[child.state, child.times_visited, child.q, child.select_probability] for child in
                              self.children.values()]
        if best_child is None:
            # This was for testing purposes
            print("Best child is None. Possibly important info:\n", self.state, "\n",
                  valids_for_state, printable_children, self.was_visited(), utcs,
                  file=open("important_info"
                            ".txt", "w"))
            # raise Exception("Best child is None,terminating.")
        return best_child, best_action

    def calculate_utc(self, c=1.5):
        if self.q is None:
            # Inspiration taken from https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
            utc = c * self.select_probability * math.sqrt(self.parent.times_visited + 1e-8)
        else:
            utc = self.q + c * (
                    self.select_probability * ((math.sqrt(self.parent.times_visited)) / (1 + self.times_visited)))

        return utc

    def get_self_value(self):
        # Not important for the algorithm, but might be useful for debugging.
        return self.total_value / self.times_visited if self.times_visited > 0 else 0

    def get_self_action_probabilities(self, tau=1.0,adjust=True):
        total_times_visited = self.times_visited
        action_probs = {}
        # action_probs = dict.fromkeys(self.children.keys(), 0)
        for action, child in self.children.items():
            action_probs[action] = child.times_visited / total_times_visited

        # any_is_zero = any([x == 0 for x in action_probs.values()])
        # any_is_zero

        # random_idx = random.randint(0, len(action_probs) - 2)
        # action_probs[random_idx] = 0.9 ** 1e-15
        # rem = 1 - (0.9 ** 1e-15)
        # action_probs[random_idx + 1] = rem
        # print(sum(action_probs.values()))

        # print(action_probs)
        if adjust:
            return GameManager.adjust_probabilities(action_probs, tau=tau)
        else:
            return action_probs
