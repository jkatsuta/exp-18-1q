import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import wanderer


class Scenario(wanderer.Scenario):
	def set_params(self):
	    self.n_agents = 1
	    self.PENALTY_WEIGHT = 0.0
	    self.VISIBLE_WEIGHT = 0.1
	    self.L_PROB = 1.0

	def get_world(self):
		world = wanderer.World3()
		world.dim_c = 3
		return world
