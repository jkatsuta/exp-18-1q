#!/usr/bin/env python
import os
import os.path as osp

exp_dir = 'simple_world_comm__2018-03-17-_01-02-02'
exp_dir = osp.join('./exp_results', exp_dir)
# n_epis = [1000, 3000, 5000]
n_epis = [60000]
num_episodes = 2
max_episode_len = 400
# outfile_suffix = 'setup2'
outfile_suffix = None

models = [osp.join(exp_dir, 'models/model-%d' % n_epi) for n_epi in n_epis]
for model in models:
    com = 'python play_trained_model.py --model %s ' % model
    com += '--num-episodes %d ' % num_episodes
    com += '--max-episode-len %d ' % max_episode_len
    if (outfile_suffix is not None) and (len(outfile_suffix) > 0):
        com += '--outfile-suffix %s ' % outfile_suffix
    os.system(com)
    # print(com)
