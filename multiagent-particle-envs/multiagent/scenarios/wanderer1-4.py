import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import wanderer


PENALTY_RATIO = 0.1
VISIBLE_RATIO = 0.1
L_PROB = 1.0


class Scenario(wanderer.Scenario):
	pass