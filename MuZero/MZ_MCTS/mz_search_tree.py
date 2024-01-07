import copy

import numpy as np
import torch as th
from bson import Binary
import pickle
import json
from Game.tictactoe_game import TicTacToeGameManager
from General.mz_game import MuZeroGame
from General.search_tree import SearchTree
from MuZero.MZ_MCTS.mz_node import MzAlphaZeroNode
from MuZero.utils import match_action_with_obs, resize_obs, scale_action
from mem_buffer import MuZeroFrameBuffer


class MuZeroSearchTree(SearchTree):

    def __init__(self, game_manager: MuZeroGame, args: dict):
        self.game_manager = game_manager
        self.args = args
        self.buffer = MuZeroFrameBuffer(self.args["frame_buffer_size"], self.game_manager.get_noop())
        self.min_max_q = [float("inf"), -float("inf")]

    def play_one_game(self, network_wrapper: th.nn.Module, device: th.device) -> tuple[list, int, int, int]:
        num_steps = self.args["num_steps"]
        frame_skip = self.args["frame_skip"]
        state = self.game_manager.reset()
        state = resize_obs(state, (96, 96))
        self.buffer.init_buffer(state)
        data = []
        for step in range(num_steps):
            pi, v, latent = self.search(network_wrapper, state, None, device)
            # print(f"Search at {step} finished",file=open("search.txt", "a"))
            move = TicTacToeGameManager.select_move(pi)
            state = match_action_with_obs(latent, move)
            _, pred_rew = network_wrapper.dynamics_forward(state.unsqueeze(0), predict=True)
            state, rew, done = self.game_manager.frame_skip_wrapper(move, None, frame_skip=frame_skip)
            state = resize_obs(state, (96, 96))
            if done:
                break
            move = scale_action(move, self.game_manager.get_num_actions())
            data.append((pi, v, (rew, move, float(pred_rew[0][0])), self.buffer.concat_frames()))
            self.buffer.add_frame(state, move)

        # data = [{"probabilities": json.dumps(pi), "vs": v, "pred_reward": pred_rew, "t_reward": rew, "game_state": Binary(pickle.dumps(state)), "game_move": move} for
        #         pi, v, (rew, move, pred_rew), state in data]
        return data, 1, 1, 1

    def search(self, network_wrapper, state: np.ndarray, current_player: int or None, device: th.device,
               tau: float or None = None):
        if len(self.buffer) == 0:
            self.buffer.init_buffer(state)
        num_simulations = self.args["num_simulations"]
        if tau is None:
            tau = self.args["tau"]

        root_node = MzAlphaZeroNode()
        # state_ = th.tensor(state, dtype=th.float32, device=device).unsqueeze(0)
        # transpose to channels first
        state_ = network_wrapper.representation_forward(
            self.buffer.concat_frames().permute(2, 0, 1).unsqueeze(0)).squeeze(0)
        pi, v = network_wrapper.prediction_forward(state_.unsqueeze(0), predict=True)
        # might mask
        pi = pi.flatten().tolist()
        root_node.expand_node(state_, pi, 0)
        for simulation in range(num_simulations):
            current_node = root_node
            path = [current_node]
            action = None
            while current_node.was_visited():
                current_node, action = current_node.get_best_child(c=self.args["c"], c2=self.args["c2"])
                path.append(current_node)

            action = scale_action(action, self.game_manager.get_num_actions())

            st = match_action_with_obs(current_node.parent.state, action)
            next_state, reward = network_wrapper.dynamics_forward(st.unsqueeze(0), predict=True)
            reward = reward[0][0]
            v = self.game_manager.game_result(current_node.current_player)
            if v is None or not v:
                pi, v = network_wrapper.prediction_forward(next_state.unsqueeze(0), predict=True)
                pi = pi.flatten().tolist()
                v = v.flatten().tolist()[0]
                current_node.expand_node(next_state, pi, reward)
            self.backprop(v, path)

        return root_node.get_self_action_probabilities(tau=tau), root_node.get_self_value(), root_node.get_latent()

    def backprop(self, v, path):
        G = v
        gamma = self.args["gamma"]
        for node in reversed(path):
            G = node.reward + gamma * G
            node.total_value += v
            node.update_q(G)
            self.update_min_max_q(node.q)
            node.scale_q(self.min_max_q[0], self.min_max_q[1])
            node.times_visited += 1

    def make_fresh_instance(self):
        return MuZeroSearchTree(self.game_manager.make_fresh_instance(), copy.deepcopy(self.args))

    def step_root(self, action: int or None):
        pass

    def update_min_max_q(self, q):
        self.min_max_q[0] = min(self.min_max_q[0], q)
        self.min_max_q[1] = max(self.min_max_q[1], q)
