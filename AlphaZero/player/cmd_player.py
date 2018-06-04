class Player:
    """
    Represents a player controlled by a human in the command line playing interface.
    """
    def __init__(self):
        pass

    def think(self, state):
        """
        Asks the user for input and returns if it's legal.

        Args:
            state: the current game state.

        Returns:
            tuple: a tuple of the input move and None.

        """
        legal = False
        while not legal:
            x, y = [int(i) for i in input('Please input the coordinate as X Y: ').split()]
            legal = state.is_legal((x, y))
        return (x, y), None

    def ack(self, move):
        """
        Does nothing.

        Args:
            move: the move played.

        Returns:
            None
        """
        pass
