import numpy as np
import torch as th
from gymnasium import make

from General.mz_game import MuZeroGame
from Game.tictactoe_game import TicTacToeGameManager


class Asteroids(MuZeroGame):
    def __init__(self):
        self.env = make("ALE/Asteroids-v5")
        self.done = False

    def reset(self) -> np.ndarray or th.Tensor:
        obs, _ = self.env.reset()
        return obs

    def get_noop(self) -> int:
        return 0

    def get_num_actions(self) -> int:
        return self.env.action_space.n

    def game_result(self, player: int or None) -> bool or None:
        return self.done

    def make_fresh_instance(self):
        return Asteroids()

    def get_next_state(self, action: int, player: int or None, frame_skip: int = 4) -> (
            np.ndarray or th.Tensor, int, bool):
        obs, rew, done, _, _ = self.env.step(action)
        self.done = done
        return obs, rew, done

    def frame_skip_wrapper(self, action: int, player: int or None, frame_skip: int = 4) -> (
            np.ndarray or th.Tensor, int, bool):
        obs, rew, done = self.get_next_state(action, player)
        for i in range(frame_skip - 1):
            obs, rew, done = self.get_next_state(action, player)
        return obs, rew, done

    def render(self):
        pass

    @staticmethod
    def select_move(action_probs: dict):
        return TicTacToeGameManager.select_move(action_probs)

    def get_random_valid_action(self,board: np.ndarray):
        return self.env.action_space.sample()
