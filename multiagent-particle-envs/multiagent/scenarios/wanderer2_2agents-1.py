import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import wanderer2


class Scenario(wanderer2.Scenario):
	def set_params(self):
	    self.n_agents = 2
	    self.PENALTY_WEIGHT = 0.0  # energy
	    self.PENALTY_WEIGHT2 = 0.0  # attention
	    self.VISIBLE_WEIGHT = 0.1
	    self.L_PROB = 1.0
