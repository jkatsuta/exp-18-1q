#!/usr/bin/env python
import os
scenarios = ['simple', 'simple_adversary', 'simple_crypto.py',
             'simple_push', 'simple_reference', 'simple_speaker_listener',
             'simple_spread', 'simple_tag', 'simple_world_comm']
num_episode = 60000

for scenario in scenarios:
    os.system('python train.py --scenario %s --num-episode %d &'
              % (scenario, num_episode))
