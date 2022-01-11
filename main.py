import numpy as np
from scipy.signal import convolve2d


class Connect4Board:

    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.board = np.zeros((self.column_count, self.row_count), dtype=int)
        self.empty_flags = np.ones(self.column_count, dtype=bool)

    def print_board(self):
        print(np.rot90(self.board))
        print()

    def drop(self, col, player):
        open_index = np.where(self.board[col] == 0)[0][0]
        self.board[col][open_index] = player
        if open_index == self.row_count - 1:
            self.empty_flags[col] = False

    def undo_drop(self, col, player):
        last_index = np.where(self.board[col] == player)[0][-1]
        self.board[col][last_index] = 0
        if last_index == self.row_count - 1:
            self.empty_flags[col] = True

    def can_drop(self, col):
        return self.empty_flags[col]

    def evaluate(self):
        h_kernel = np.array([[1, 1, 1, 1]])
        v_kernel = np.transpose(h_kernel)
        d1_kernel = np.eye(4, dtype=int)
        d2_kernel = np.eye(4, dtype=int)[:][::-1]
        win_kernels = [h_kernel, v_kernel, d1_kernel, d2_kernel]

        p1 = self.board == 1
        p2 = self.board == 2

        for kernel in win_kernels:
            if (convolve2d(p1, kernel, mode="valid") == 4).any():
                return 1000
            if (convolve2d(p2, kernel, mode="valid") == 4).any():
                return -1000
        return 0

    def moves_left(self):
        return self.empty_flags.any()


def main():
    board = Connect4Board()


if __name__ == '__main__':
    main()
