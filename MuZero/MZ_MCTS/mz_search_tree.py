import numpy as np
import torch as th

from Game.game import Game
from MuZero.MZ_MCTS.mz_node import MzNode
from mem_buffer import MuZeroFrameBuffer
from MuZero.Network.networks import MuZeroNet


class MuZeroSearchTree:
    def __init__(self, game_manager: Game, args: dict):
        self.game_manager = game_manager
        self.args = args
        self.buffer = MuZeroFrameBuffer(self.args["frame_buffer_size"])

    def simulate(self, network_wrapper, device: th.device):
        num_steps = self.args["num_steps"]
        frame_skip = self.args["frame_skip"]
        state = self.game_manager.reset()

    def search(self, network_wrapper: MuZeroNet, state: np.ndarray, device: th.device, tau: float = None):
        num_simulations = self.args["num_simulations"]
        if tau is None:
            tau = self.args["tau"]

        root_node = MzNode()
        state_ = th.tensor(state,dtype=th.float32,device=device).unsqueeze(0)
        self.buffer.init_buffer(state_)
        cat = self.buffer.concat_frames()
        state_ = network_wrapper.representation_forward(cat)
        pi, v = network_wrapper.prediction_forward(cat,predict=True)
        # might mask
        pi = pi.flatten().tolist()
        root_node.expand(state_,pi)
        for simulation in range(num_simulations):
            current_node = root_node
            path = [current_node]
            action = None
            while current_node.was_visited():
                current_node,action = current_node.get_best_child(c=self.args["c"],c2=self.args["c2"])
                path.append(current_node)

            next_state = network_wrapper.dynamics_forward(current_node.parent.state,action)
            




