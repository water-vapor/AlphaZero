import sgf
import AlphaGo.settings as s

def _convert_sgf_move(move):
    """Converts a move of {'B': ['ab']} type to ('B',[0, 1]).
    
    Args:
        move (dict): A dictionary with player as key, and a positional string.
        
    Returns:
        tuple: player and a list of position coordinates.
    """
    for player, [position] in move.items():
        #pass move
        if position == '' or position == 'tt':
            return player, [-1, -1]
        #ord(a)==97, ord(coord)-97 is the index starting from 0
        return player, [ord(coord)-97 for coord in position]


def _sgf_move_generator(game):
    """Generate moves from a GameTree object.
    
    Args:
        game (GameTree): GameTree object from the collection of parsed sgf.
        
    Yields:
        tuple: player and a list of position coordinates for every move in a game.
    """
    game_info = game.root.properties
    if 'AW' in game_info:
        for position in game_info['AW']:
            yield _convert_sgf_move({'W': [position]})
    if 'AB' in game_info:
        for position in game_info['AB']:
            yield _convert_sgf_move({'B': [position]})
    for node in game.rest:
        move = node.properties
        yield _convert_sgf_move(move)



