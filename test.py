# Test pieces of code

import numpy as np
import torch
from Game.game import GameManager
import torch.nn.functional as F
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


def softmax_logsoftmax_test():
    alpha = 3000000
    true_result = torch.log(torch.softmax(alpha * torch.tensor([-1.0, 0.0, 1.0]).double(), dim=0))
    res1 = torch.log(torch.softmax(alpha * torch.tensor([-1.0, 0.0, 1.0]), dim=0))
    # and
    res2 = F.log_softmax(alpha * torch.tensor([-1.0, 0.0, 1.0]), dim=0)
    print(f"True result: {true_result}")
    print("\n-------------\n")
    print(f"log(sotfmax(x)) result: {res1}")
    print(f"log_softmax(x) result: {res2}")
    print(f"log_softmax(x) reversed: {torch.exp(res2)}")
    print(f"log_softmax(x) reversed: {torch.exp(res2)}")
    print(torch.softmax(alpha * torch.tensor([-1.0, 0.0, 1.0]),dim=-1))


if __name__ == "__main__":
    softmax_logsoftmax_test()
    # print(num_of_tic_tac_toe_winning_states(5,3)/ num_of_tic_tac_toe_winning_states(5,5))
