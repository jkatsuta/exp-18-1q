#!/usr/bin/env python
import os
import os.path as osp

exp_dir = 'exp_simple_tag_14-03-2018_05-03-56'
exp_dir = osp.join('./exp_results/gpu_jk', exp_dir)
# n_epis = [1000, 3000, 5000]
n_epis = [60000]
num_episodes = 2
max_episode_len = 400

models = [osp.join(exp_dir, 'models/model-%d' % n_epi) for n_epi in n_epis]
for model in models:
    com = 'python play_trained_model.py --model %s ' % model
    com += '--num-episodes %d ' % num_episodes
    com += '--max-episode-len %d ' % max_episode_len
    os.system(com)
    # print(com)
