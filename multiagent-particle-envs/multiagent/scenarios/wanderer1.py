import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import wanderer


class Scenario(wanderer.Scenario):
	def get_world(self):
		return wanderer.World2()
