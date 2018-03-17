#!/usr/bin/env python
import os
import os.path as osp

# exp_dir = 'simple_tag__2018-03-16-_19-15-55'
exp_dir = 'simple_tag__2018-03-16-_21-44-54/'
exp_dir = osp.join('./exp_results', exp_dir)
# n_epis = [1000, 3000, 5000]
n_epis = [10000, 20000, 40000]
num_episodes = 1
max_episode_len = 400

models = [osp.join(exp_dir, 'models/model-%d' % n_epi) for n_epi in n_epis]
for model in models:
    com = 'python play_trained_model.py --model %s ' % model
    com += '--num-episodes %d ' % num_episodes
    com += '--max-episode-len %d ' % max_episode_len
    os.system(com)
    # print(com)
