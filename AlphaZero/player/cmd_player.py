class Player:
    def __init__(self):
        pass

    def think(self, state):
        legal = False
        while not legal:
            x, y = [int(i) for i in input('Please input the coordinate as X Y: ').split()]
            legal = state.is_legal((x, y))
        return (x, y), None

    def ack(self, move):
        pass
