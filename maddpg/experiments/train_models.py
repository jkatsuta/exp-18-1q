#!/usr/bin/env python
import os
import sys

params = eval(open(sys.argv[1]).read())

num_episodes = params['num_episodes']
max_episode_lens =\
    params.get('max_episode_lens', [None] * len(num_episodes))

for scenario in params['scenarios']:
    for i, num_episode in enumerate(num_episodes):
        com = 'python train.py --scenario %s --num-episodes %d '\
            % (scenario, num_episode)
        if max_episode_lens[i] is not None:
            com += '--max-episode-len %s ' % max_episode_lens[i]
        if params['is_parallel']:
            com += ' &'
        os.system(com)
        # print(com)
