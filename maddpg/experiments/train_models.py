#!/usr/bin/env python
import os
import sys

params = eval(open(sys.argv[1]).read())

for scenario in params['scenarios']:
    for i, num_episode in enumerate(params['num_episodes']):
        com = 'python train.py --scenario %s ' % scenario
        com += '--num-episodes %d ' % num_episode
        if params.get('is_variable_max_episode_len', False):
            com += '--dic-variable-max-episode-len "%s" '\
                % str(params['par_variable_max_episode_len'])
        else:
            com += '--max-episode-len %s ' % params['max_episode_lens'][i]
        if params['is_parallel']:
            com += ' &'
        os.system(com)
        # print(com)
