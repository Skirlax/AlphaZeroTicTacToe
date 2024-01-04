import random
from typing import Type

import torch as th

from AlphaZero.Arena.players import Player
from AlphaZero.utils import DotDict
from General.arena import GeneralArena
from General.mz_game import MuZeroGame
from MuZero.utils import resize_obs


class MzArena(GeneralArena):
    def __init__(self, game_manager: MuZeroGame, args: DotDict, device: th.device):
        self.game_manager = game_manager
        self.args = args
        self.device = device

    def pit(self, player1: Type[Player], player2: Type[Player], num_games_to_play: int, num_mc_simulations: int,
            one_player: bool = False, start_player: int = 1):
        tau = self.args.arena_tau
        rewards = {1: [], -1: []}
        if one_player:
            num_games_per_player = num_games_to_play
        else:
            num_games_per_player = num_games_to_play // 2
        for player in [1, -1]:
            kwargs = {"num_simulations": num_mc_simulations, "current_player": player, "device": self.device,
                      "tau": tau, "unravel": False}
            for game in range(num_games_per_player):
                self.game_manager.reset()
                noop_num = random.randint(0, 30)
                state, _, _ = self.game_manager.frame_skip_wrapper(self.game_manager.get_noop(), None,
                                                                   frame_skip=noop_num)
                state = resize_obs(state, (96, 96))
                for step in range(self.args["num_steps"]):
                    self.game_manager.render()
                    if player == 1:
                        move = player1.choose_move(state, **kwargs)
                    else:
                        move = player2.choose_move(state, **kwargs)
                    state, reward, done = self.game_manager.frame_skip_wrapper(move, None)
                    state = resize_obs(state, (96, 96))
                    rewards[player].append(reward)
                    if done:
                        break

        return sum(rewards[1]), sum(rewards[-1]), 0
