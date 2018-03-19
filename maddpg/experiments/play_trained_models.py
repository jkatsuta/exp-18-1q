#!/usr/bin/env python
import os
import os.path as osp


def play_trained_model(model, num_episodes, max_episode_len, outfile_suffix,
                       display_speed=None):
    com = 'python play_trained_model.py --model %s ' % model
    com += '--num-episodes %d ' % num_episodes
    com += '--max-episode-len %d ' % max_episode_len
    if (outfile_suffix is not None) and (len(outfile_suffix) > 0):
        com += '--outfile-suffix %s ' % outfile_suffix
    if display_speed is not None:
        com += '--display-speed %s ' % display_speed
    os.system(com)
    # print(com)


if __name__ == '__main__':
    exp_dir = 'simple_tag__2018-03-16-_19-15-55'
    # exp_dir = 'simple_tag__2018-03-16-_21-44-54'
    # exp_dir = 'simple_tag__2018-03-16-_19-15-55'
    exp_dir = osp.join('./exp_results', exp_dir)
    # n_epis = [10000, 20000, 40000]
    n_epis = [60000]
    num_episodes = 2
    max_episode_len = 400
    outfile_suffix = 'epi-len400'
    # outfile_suffix = None
    display_speed = 'high' #'slow'

    models = [osp.join(exp_dir, 'models/model-%d' % n_epi) for n_epi in n_epis]
    for model in models:
        play_trained_model(model, num_episodes, max_episode_len, outfile_suffix,
                           display_speed)
