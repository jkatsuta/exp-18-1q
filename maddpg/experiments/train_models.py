#!/usr/bin/env python
import os
import sys

params = eval(open(sys.argv[1]).read())

num_episodes = params['num_episodes']
max_episode_lens = params['max_episode_lens']

for scenario in params['scenarios']:
    for num_episode, max_episode_len in zip(num_episodes, max_episode_lens):
        com = 'python train.py --scenario %s --num-episodes %d '\
            % (scenario, num_episode)
        com += '--max-episode-len %d ' % max_episode_len
        if params['is_parallel']:
            com += ' &'
        os.system(com)
        #print(com)
