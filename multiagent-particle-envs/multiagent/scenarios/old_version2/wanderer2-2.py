import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import wanderer


class Scenario(wanderer.Scenario):
	def set_params(self):
	    self.dim_c = 2  # 0: borrow money, 1: do nothing
	    self.n_agents = 2
	    self.PENALTY_RATIO = 0.01
	    self.VISIBLE_RATIO = 0.1
	    self.L_PROB = 1.0
