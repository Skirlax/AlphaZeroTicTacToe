import numpy as np
from manimlib import Scene, Circle, Dot, Line, Square, Tex, VGroup
from manimlib import GOLD, BLUE, BLUE_E, BLACK, RED


class TicTacToeBoard(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.board = np.zeros((3, 3))
        self.board[0][2] = 1
        self.board[0, 1] = -1
        self.board = np.flip(self.board, axis=1)

    def construct(self) -> None:
        self.draw_board()
        self.draw_pieces()

    def draw_board(self) -> None:
        self.board_squares = VGroup()
        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                square = Square(side_length=1, color=BLUE_E)
                square.move_to(np.array([row, col, 0]))
                self.board_squares.add(square)
        self.add(self.board_squares)

    def draw_pieces(self) -> None:
        self.pieces = VGroup()
        half_line_size = 0.3
        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                if self.board[row, col] == 1:
                    piece = Circle(radius=0.3, color=GOLD)
                    piece.move_to(np.array([row, col, 0]))
                    self.pieces.add(piece)
                elif self.board[row, col] == -1:
                    piece = Line(start=np.array([row - half_line_size, col - half_line_size, 0]), end=np.array([row + half_line_size, col + half_line_size, 0]),
                                 color=RED)
                    piece.move_to(np.array([row, col, 0]))
                    self.pieces.add(piece)
                    piece = Line(start=np.array([row - half_line_size, col + half_line_size, 0]), end=np.array([row + half_line_size, col - half_line_size, 0]),
                                 color=RED)
                    piece.move_to(np.array([row, col, 0]))
                    self.pieces.add(piece)

        self.add(self.pieces)
