#!/usr/bin/env python
import os
import os.path as osp

exp_dir = 'exp_simple_world_comm_13-03-2018_16-09-17'
exp_dir = osp.join('./exp_results/gmoapp', exp_dir)
# n_epis = [30000]
n_epis = [3000, 10000, 30000, 60000]

models = [osp.join(exp_dir, 'models/model-%d' % n_epi) for n_epi in n_epis]
for model in models:
    com = 'python play_trained_model.py --model %s ' % model
    os.system(com)
    # print(com)
