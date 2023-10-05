import time

from AlphaZero.Arena.players import NetPlayer, Player
from AlphaZero.utils import DotDict
from Game.game import GameManager
from typing import Type


class Arena:
    def __init__(self, game_manager: GameManager, args: DotDict, device):
        self.game_manager = game_manager
        self.device = device
        self.args = args

    def pit(self, player1: Type[Player], player2: Type[Player], num_games_to_play: int, num_mc_simulations: int):
        """
        Pit two players against each other for a given number of games and gather the results.
        :param player1:
        :param player2:
        :param num_games_to_play:
        :param num_mc_simulations:
        :return: number of wins for player1, number of wins for player2, number of draws
        """
        results = {"wins_p1": 0, "wins_p2": 0, "draws": 0}
        tau = self.args.arena_tau
        num_games_per_player = num_games_to_play // 2

        for game in range(num_games_to_play):
            if game < num_games_per_player:
                current_player = 1
            else:
                current_player = -1
            kwargs = {"num_simulations": num_mc_simulations, "current_player": current_player, "device": self.device,
                      "tau": tau}
            state = self.game_manager.reset()
            player1.monte_carlo_tree_search.step_root(None)
            if player2.name == "NetworkPlayer":
                player2.monte_carlo_tree_search.step_root(None)

            while True:
                self.game_manager.render()
                if current_player == 1:
                    move = player1.choose_move(state, **kwargs)
                    self.game_manager.play(current_player, move)
                else:
                    move = player2.choose_move(state, **kwargs)
                    self.game_manager.play(current_player, move)
                state = self.game_manager.get_board()
                status = self.game_manager.game_result(current_player, state)
                self.game_manager.render()
                if status is not None:
                    if status == 1:
                        if current_player == 1:
                            results["wins_p1"] += 1
                        else:
                            results["wins_p2"] += 1

                    elif status == -1:
                        if current_player == 1:
                            results["wins_p2"] += 1
                        else:
                            results["wins_p1"] += 1
                    else:
                        results["draws"] += 1

                    # time.sleep(2)
                    break

                current_player *= -1
                kwargs["current_player"] = current_player

        return results["wins_p1"], results["wins_p2"], results["draws"]
