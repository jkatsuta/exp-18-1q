#!/usr/bin/env python
import os
import sys
import re
import time


def _train_model(i_epi, scenario, num_episode, pars, dic_var_epi_len,
                 seed=None, test_mode=False):
    com = 'python train.py --scenario %s ' % scenario
    com += '--num-episodes %d ' % num_episode
    if len(dic_var_epi_len) > 0:
        com += '--dic-variable-max-episode-len "%s" ' % str(dic_var_epi_len)
    else:
        if 'max_episode_lens' in pars.keys():
            com += '--max-episode-len %s ' % pars['max_episode_lens'][i_epi]
    if 'good_policy' in pars.keys():
        com += '--good-policy %s ' % pars['good_policy']
    if 'adv_policy' in pars.keys():
        com += '--adv-policy %s ' % pars['adv_policy']
    if pars.get('restored_model', False):
        com += '--restore --load-model %s ' % pars['restored_model']
        seed = re.search('\_seed(\d+)/models/', pars['restored_model'])
        if seed and not pars.get('seed', False):
            com += '--seed %d ' % int(seed.group(1))
    if seed is not None:
        com += '--seed %d ' % seed
    if pars['is_parallel']:
        com += ' &'
    print(com)
    if not test_mode:
        os.system(com)
        time.sleep(2)


def get_dics_pars(fn_param):
    dics_pars = eval(open(fn_param).read())
    if isinstance(dics_pars, dict):
        dics_pars = [dics_pars]
    return dics_pars


def get_seeds(pars):
    seeds = pars.get('seeds', [None])
    if isinstance(seeds, int):
        seeds = [seeds]
    return seeds


def train_models(pars, seed=None, test_mode=False):
    dic_var_epi_lens = [{}]
    if pars.get('is_variable_max_episode_len', False):
        dic_var_epi_lens = pars['par_variable_max_episode_lens']

    for scenario in pars['scenarios']:
        for dic_var_epi_len in dic_var_epi_lens:
            for i, num_episode in enumerate(pars['num_episodes']):
                _train_model(i, scenario, num_episode, pars,
                             dic_var_epi_len, seed, test_mode)


if __name__ == '__main__':
    fn_param = sys.argv[1]
    try:
        test_mode = eval(sys.argv[2])
    except IndexError:
        test_mode = False

    dics_pars = get_dics_pars(fn_param)
    for dic_par in dics_pars:
        seeds = get_seeds(dic_par)
        for seed in seeds:
            train_models(dic_par, seed, test_mode)
