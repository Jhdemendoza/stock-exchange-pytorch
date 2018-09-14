import random
import numpy as np


class Engine:
    """ 2048 Game class """

    def __init__(self, num_observed_tickers=4, state=None, seed=None):
        # Number of moves available
        self.N = num_observed_tickers
        self.reset_game(state)
        if seed:
            random.seed(seed)

    def reset_game(self, state=None):
        self.score = 0
        self.ended = False
        self.won = False
        self.last_move = '-'

        self.state = state

        self.board = [[0]*self.N for i in range(self.N)]
        self.merged = [[False]*self.N for i in range(self.N)]

    # Returns state
    def get_board(self):
        return self.board


    def find_furthest(self, row, col, vector):
        """ finds furthest cell interactable (empty or same value) """
        found = False
        val = self.board[row][col]
        i = row + vector['y']
        j = col + vector['x']
        while i >= 0 and i < self.N and j >= 0 and j < self.N:
            val_tmp = self.board[i][j]
            if self.merged[i][j] or (val_tmp != 0 and val_tmp != val):
                return (i - vector['y'], j - vector['x'])
            if val_tmp:
                return (i, j)

            i += vector['y']
            j += vector['x']

        return (i - vector['y'], j - vector['x'])

    def moves_available(self):
        raise NotImplementedError

    # make a move
    def move(self, direction):
        # up: 0, right: 1, down: 2, left: 3

        if moved:
            self.add_random()

        self.ended = not True in self.moves_available() # or self.won
        if self.ended and not self.won:
            logged_reward = -3.0
        else:
            logged_reward = np.clip(np.log2(reward), 0, 18) / 11.0 # log2(1024) == 10, 18 totally aribtrary

        return logged_reward, self.ended

    def __str__(self):
        max_len = len(str(max(max(self.board))))
        board_str = ""
        for row in self.board:
            padded_row = [str(cell).rjust(max_len) for cell in row]
            board_str += "{0} {1} {2} {3}\n".format(*padded_row)

        board_str += "Score: {}\n".format(self.score)
        board_str += "Move: {}\n".format(self.last_move)
        return board_str

