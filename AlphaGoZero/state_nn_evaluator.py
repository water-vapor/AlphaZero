import numpy as np
import random
from AlphaGoZero import go

def random_state_transform(state):
	""" Performs a dihedral reflection or rotation.
	"""
	transformed_state = state.copy()
	transform_id = random.randint(0, 7)
	transformed_state.transform(transform_id)
	return transformed_state

# TODO: Implement parallel manager to send a batch of 8 states to NN

