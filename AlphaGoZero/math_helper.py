import numpy as np
import random

def random_state_transform(state):
	""" Performs a dihedral reflection or rotation.
	"""
	transformed_state = state.copy()
	transform_id = random.randint(0, 7)
	transformed_state.transform(transform_id)
	return transformed_state

def random_variate_dirichlet(alpha, length):
	pass

def weighted_random_choice(li):
	# This requires Python 3.6, should be implemented otherwise
	p, w = zip(*li)
	return random.choices(p, weights=w)[0]


