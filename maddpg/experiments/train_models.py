#!/usr/bin/env python
import os
import sys

params = eval(open(sys.argv[1]).read())

for scenario in params['scenarios']:
    com = 'python train.py --scenario %s --num-episodes %d'\
        % (scenario, params['num_episodes'])
    if params['is_parallel']:
        com += ' &'
    os.system(com)
    # print(com)
