import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import wanderer1


class Scenario(wanderer1.Scenario):
	def set_params(self):
	    self.n_agents = 4
	    self.PENALTY_WEIGHT = 0.0
	    self.VISIBLE_WEIGHT = 0.1
	    self.L_PROB = 1.0
