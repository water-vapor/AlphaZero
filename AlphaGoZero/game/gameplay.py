import AlphaGoZero.game.player as player
import AlphaGoZero.go as go

class Game:

    def __init__(self, nn_eval_1, nn_eval_2):
        self.player_1 = player.Player(nn_eval_1)
        self.player_2 = player.Player(nn_eval_2)
        self.state = go.GameState()

    def __call__(self, *args, **kwargs):
        pass_1 = pass_2 = False
        while not (pass_1 and pass_2):
            move = self.player_1.think(self.state)
            if move is go.PASS_MOVE:
                pass_1 = True # TODO: check the type of variable move
            self.state.do_move(move)
            self.player_1.observe(move)
            self.player_2.observe(move)

            move = self.player_2.think(self.state)
            if move is go.PASS_MOVE:
                pass_2 = True
            self.state.do_move(move)
            self.player_1.observe(move)
            self.player_2.observe(move)

            # TODO: other end game condition

        return self.state.get_winner()
