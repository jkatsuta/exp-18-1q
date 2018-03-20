#!/usr/bin/env python
import os
import sys
import os.path as osp


def play_trained_model(model, pars, test_mode=False):
    com = 'python play_trained_model.py --model %s ' % model
    com += '--num-episodes %d ' % pars['num_episodes']
    com += '--max-episode-len %d ' % pars['max_episode_len']
    if (pars['outfile_suffix'] is not None) and (len(pars['outfile_suffix']) > 0):
        com += '--outfile-suffix %s ' % pars['outfile_suffix']
    if pars['display_speed'] is not None:
        com += '--display-speed %s ' % pars['display_speed']
    if 'good_policy' in pars.keys():
        com += '--good-policy %s ' % pars['good_policy']
    if 'adv_policy' in pars.keys():
        com += '--adv-policy %s ' % pars['adv_policy']

    if test_mode:
        print(com)
    else:
        os.system(com)


if __name__ == '__main__':
    fn_par = sys.argv[1]
    try:
        test_mode = eval(sys.argv[2])
    except IndexError:
        test_mode = False

    dics_pars = eval(open(fn_par).read())
    for pars in dics_pars:
        for n_epi in pars['n_epis']:
            model = osp.join(pars['p_dir'], pars['exp_dir'],
                             'models/model-%d' % n_epi)
            play_trained_model(model, pars, test_mode)
            # break
        # break
