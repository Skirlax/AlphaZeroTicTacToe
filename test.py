# Test pieces of code

import numpy as np

from Game.game import GameManager

manager = GameManager(3, True)





def brute_force_winning_states(n: int, k: int):
    states_num = 3 ** (n ** 2)
    winning_states = 0
    for i in range(states_num):
        board = np.zeros((n, n), dtype=np.int8)
        p = i
        for row in range(n):
            for col in range(n):
                x = p % 3
                if x == 0:
                    board[row][col] = 0
                elif x == 1:
                    board[row][col] = 1
                else:
                    board[row][col] = -1
                p //= 3

        if manager.check_partial_win(1, k, board=board) or manager.check_partial_win(-1, k, board=board):
            winning_states += 1
        # del board

    return winning_states


if __name__ == "__main__":
    pass
    # print(num_of_tic_tac_toe_winning_states(5,3)/ num_of_tic_tac_toe_winning_states(5,5))
