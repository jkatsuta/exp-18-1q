#!/usr/bin/env python
import os
import sys
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
    fn_par = sys.argv[1]
    dics_pars = eval(open(fn_par).read())

    for pars in dics_pars:
        for n_epi in pars['n_epis']:
            model = osp.join(pars['p_dir'], pars['exp_dir'],
                             'models/model-%d' % n_epi)
            play_trained_model(model,
                               pars['num_episodes'],
                               pars['max_episode_len'],
                               pars['outfile_suffix'],
                               pars['display_speed'])
            # break
        # break
