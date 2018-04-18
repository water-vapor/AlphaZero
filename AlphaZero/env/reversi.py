from copy import deepcopy

import numpy as np

WHITE = -1
BLACK = +1
PASS_MOVE = None
EMPTY = 0


class GameState(object):
    """ Game state of Reversi Game.

    """

    def __init__(self, size=8, history_length=8):
        # the size of reversi should not be changed
        self.board = np.zeros((size, size), dtype=int)
        self.board.fill(EMPTY)
        self.board[size // 2 - 1][size // 2 - 1] = WHITE
        self.board[size // 2][size // 2] = WHITE
        self.board[size // 2][size // 2 - 1] = BLACK
        self.board[size // 2 - 1][size // 2] = BLACK
        self.size = size
        self.height = size
        self.width = size
        self.current_player = BLACK
        self.history = []
        # Keeps 8 history board for fast feature extraction
        # Fill zeros for non-existing histories
        # board_history does not include the current board while the feature does,
        self.history_length = history_length
        self.board_history = [np.zeros((size, size), dtype=int) for _ in range(history_length - 1)]
        self.is_end_of_game = False
        self.stones_played = 4
        self.turns = 0

    def _on_board(self, position):
        """simply return True iff position is within the bounds of [0, self.size)
        """
        (x, y) = position
        return x >= 0 and y >= 0 and x < self.size and y < self.size

    def _legal_direction(self, action, direction, do_move=False):
        """ A function called by is_legal, check only one direction
        
        Args:
            direction: (a, b) where a, b is -1, 0 or 1, total 8 possible directions (0, 0) is not valid
            do_move: will flip opponent's stones if valid

        Returns: boolean

        """
        x, y = action
        dx, dy = direction
        x, y = x + dx, y + dy
        # at least flip one opponent's stone
        if (not self._on_board((x, y))) or self.board[x][y] != -self.current_player:
            return False
        # skip all continuous opponent stones
        while self._on_board((x, y)) and self.board[x][y] == -self.current_player:
            x, y = x + dx, y + dy
        # opponent stones reaches boundary, cannot flip
        if not self._on_board((x, y)):
            return False
        # can flip
        if self.board[x][y] == self.current_player:
            if do_move:
                # travels back to action point and flip all stones
                while (x, y) != action:
                    self.board[x][y] = self.current_player
                    x, y = x - dx, y - dy
            return True
        # position is empty, cannot flip
        else:
            return False

    def copy(self):
        """get a copy of this Game state
        """
        other = GameState(self.size)
        other.board = self.board.copy()
        other.current_player = self.current_player
        other.history = list(self.history)
        other.board_history = deepcopy(self.board_history)
        other.turns = self.turns
        other.is_end_of_game = self.is_end_of_game
        other.stones_played = self.stones_played
        return other

    def is_legal(self, action):
        """determine if the given action (x,y) is a legal move
            Note that PASS should be a forced move in Reversi (only when there is no legal move)
            When a PASS is received, the function will check globally with get_legal_moves
        """
        # PASS is only legal when there is no legal move
        if action is PASS_MOVE:
            return self.get_legal_moves() == []

        (x, y) = action
        if not self._on_board(action):
            return False
        if self.board[x][y] != EMPTY:
            return False
        if self._legal_direction(action, (-1, -1)) or self._legal_direction(action, (-1, 0)) or \
                self._legal_direction(action, (-1, 1)) or self._legal_direction(action, (0, -1)) or \
                self._legal_direction(action, (0, 1)) or self._legal_direction(action, (1, -1)) or \
                self._legal_direction(action, (1, 0)) or self._legal_direction(action, (1, 1)):
            return True
        else:
            return False

    def get_legal_moves(self):
        """ This function is infrequently used, therefore not optimized.
            Checks all non-pass moves

        Returns: a list of legal moves

        """
        legal_moves = [(i, j) for i in range(self.size) for j in range(self.size) if self.is_legal((i, j))]
        return legal_moves

    def get_winner(self):
        """ Counts the stones on the board, assumes the game is ended

        Returns: The winner, None if the game is not ended yet

        """
        black_count = (self.board == BLACK).sum()
        white_count = (self.board == WHITE).sum()
        if black_count > white_count:
            winner = BLACK
        elif black_count < white_count:
            winner = WHITE
        else:
            winner = 0
        return winner

    def do_move(self, action, color=None):
        """Play stone at action=(x,y). If color is not specified, current_player is used
        If it is a legal move, current_player switches to the opposite color
        If not, an IllegalMove exception is raised
        """
        color = color or self.current_player
        reset_player = self.current_player
        self.current_player = color
        if self.is_legal(action):
            # save current board to history before it is modified
            self.board_history.append(self.board.copy())
            self.board_history.pop(0)
            self.history.append(action)
            self.turns += 1

            # do action
            if action is not PASS_MOVE:
                (x, y) = action
                self.board[x][y] = color
                self.stones_played += 1

                # flip stones
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        self._legal_direction(action, (di, dj), do_move=True)

                # check if stone has filled the board if no one wins yet
                if self.stones_played == self.size * self.size:
                    self.is_end_of_game = True

            else:
                if self.history[-2] == PASS_MOVE:
                    self.is_end_of_game = True

            # next turn
            self.current_player = -color

        else:
            self.current_player = reset_player
            raise IllegalMove(str(action))

        return self.is_end_of_game

    def transform(self, transform_id):
        """ Transform the current board and the history boards according to D(4).
            Caution: self.history (action history) is not modified, thus this function
            should ONLY be used for state evaluation.
            Arguments:
                transform_id: integer in range [0, 7]
        """
        def _transform(b):
            # Performs reflection
            if transform_id // 4 == 1:
                b = np.fliplr(b)
            # Performs rotation
            b = np.rot90(b, transform_id % 4)
            return b
        # List of boards to transform
        self.board = _transform(self.board)
        self.board_history = [_transform(b) for b in self.board_history]


class IllegalMove(Exception):
    pass
