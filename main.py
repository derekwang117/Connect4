import numpy as np
from scipy.signal import convolve2d

import time


def other(this):
    if this == 1:
        return 2
    return 1


def check_symmetric(board):
    return np.array_equal(board, np.flipud(board))


class Connect4Board:

    def __init__(self):
        self.row_count = 4
        self.column_count = 5
        self.board = np.zeros((self.column_count, self.row_count), dtype=int)
        self.empty_flags = np.ones(self.column_count, dtype=bool)
        self.col_flag = np.zeros(self.column_count, dtype=int)

        self.ZobristTable = np.random.randint(2147483647, 9223372036854775807,
                                              size=(self.column_count, self.row_count, 2), dtype=np.int64)
        self.hash = self.compute_hash()
        print(self.hash)

    def compute_hash(self):
        hash_val = 0
        for i in range(self.column_count):
            for j in range(self.row_count):
                if self.board[i][j]:
                    hash_val ^= self.ZobristTable[i][j][self.board[i][j] - 1]
        return hash_val

    def update_hash(self, col, row, changed):
        self.hash ^= self.ZobristTable[col][row][changed - 1]

    def print_board(self):
        print(np.rot90(self.board))
        print()

    def drop(self, col, player):
        open_index = np.where(self.board[col] == 0)[0][0]
        self.board[col][open_index] = player
        if open_index == self.row_count - 1:
            self.empty_flags[col] = False
        self.col_flag[col] += 1

        self.update_hash(col, open_index, player)
        print(self.hash)

    def undo_drop(self, col, player):
        last_index = np.where(self.board[col] == player)[0][-1]
        self.board[col][last_index] = 0
        if last_index == self.row_count - 1:
            self.empty_flags[col] = True
        self.col_flag[col] -= 1

        self.update_hash(col, last_index, player)
        print(self.hash)

    def can_drop(self, col):
        return self.empty_flags[col]

    def evaluate(self, player):
        h_kernel = np.array([[1, 1, 1, 1]])
        v_kernel = np.transpose(h_kernel)
        d1_kernel = np.eye(4, dtype=int)
        d2_kernel = np.eye(4, dtype=int)[:][::-1]
        win_kernels = [h_kernel, v_kernel, d1_kernel, d2_kernel]

        b_player = self.board == player
        b_other = self.board == other(player)

        for kernel in win_kernels:
            if (convolve2d(b_player, kernel, mode="valid") == 4).any():
                return 1000
            if (convolve2d(b_other, kernel, mode="valid") == 4).any():
                return -1000
        return 0

    def moves_left(self):
        return self.empty_flags.any()

    def center_order_empty_flags(self, is_symmetric):
        empty_indices = np.where(self.empty_flags)[0]
        if is_symmetric:
            empty_indices = np.where(np.array_split(self.empty_flags, 2)[0])[0]
        ordered_indices = np.argsort((empty_indices - (self.column_count - 1) / 2) ** 2 +
                                     (self.col_flag[empty_indices] - (self.row_count - 1) / 2) ** 2)
        return empty_indices[ordered_indices]

    def minimax(self, depth, is_max, bot_number, alpha, beta, is_symmetric):
        score = self.evaluate(bot_number)

        if score == 1000:
            return score - depth
        if score == -1000:
            return score + depth
        if not self.moves_left():
            return 0

        if is_max:
            best_val = -1001

            for pos_move in self.center_order_empty_flags(is_symmetric):
                self.drop(pos_move, bot_number)
                if is_symmetric:
                    new_val = self.minimax(depth + 1, not is_max, bot_number, alpha, beta, check_symmetric(self.board))
                else:
                    new_val = self.minimax(depth + 1, not is_max, bot_number, alpha, beta, False)
                self.undo_drop(pos_move, bot_number)

                if new_val > best_val:
                    best_val = new_val
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break

            return best_val

        else:
            best_val = 1001

            for pos_move in self.center_order_empty_flags(is_symmetric):
                self.drop(pos_move, other(bot_number))
                if is_symmetric:
                    new_val = self.minimax(depth + 1, not is_max, other(bot_number), alpha, beta, check_symmetric(self.board))
                else:
                    new_val = self.minimax(depth + 1, not is_max, other(bot_number), alpha, beta, False)
                self.undo_drop(pos_move, other(bot_number))

                if new_val < best_val:
                    best_val = new_val
                beta = min(beta, best_val)
                if beta <= alpha:
                    break

            return best_val

    def find_best_move(self, bot_number):
        best_val = -1001
        best_move = -1

        for pos_move in self.center_order_empty_flags(check_symmetric(self.board)):
            self.drop(pos_move, bot_number)
            move_val = self.minimax(0, False, bot_number, -1001, 1001, check_symmetric(self.board))
            self.undo_drop(pos_move, bot_number)

            if move_val > best_val:
                best_val = move_val
                best_move = pos_move

        print(best_move)
        return best_move


def fill_board(board, move_sequence):
    for index, char in enumerate(move_sequence):
        if index % 2 == 0:
            board.drop(int(char), 1)
        else:
            board.drop(int(char), 2)


def test(board):
    start_time = round(time.time() * 1000)

    bot_n = 2
    x = board.find_best_move(bot_n)
    board.drop(x, bot_n)
    board.print_board()

    end_time = round(time.time() * 1000)
    print("Time: " + str(end_time - start_time) + " ms")


def main():
    c4_game = Connect4Board()

    fill_board(c4_game, "222")
    c4_game.print_board()

    # zobrist hashing and multithreading
    #test(c4_game)


if __name__ == '__main__':
    main()
