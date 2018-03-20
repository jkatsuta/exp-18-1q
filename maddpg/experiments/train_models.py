#!/usr/bin/env python
import os
import sys


def _train_model(i_epi, scenario, num_episode, params, dic_var_epi_len,
                 test_mode=False):
    com = 'python train.py --scenario %s ' % scenario
    com += '--num-episodes %d ' % num_episode
    if len(dic_var_epi_len) > 0:
        com += '--dic-variable-max-episode-len "%s" ' % str(dic_var_epi_len)
    else:
        if 'max_episode_lens' in params.keys():
            com += '--max-episode-len %s ' % params['max_episode_lens'][i_epi]
    if 'good_policy' in params.keys():
        com += '--good-policy %s ' % params['good_policy']
    if 'adv_policy' in params.keys():
        com += '--adv-policy %s ' % params['adv_policy']
    if params.get('restored_model', False):
        com += '--restore --load-model %s ' % params['restored_model']
    if params['is_parallel']:
        com += ' &'
    print(com)
    if not test_mode:
        os.system(com)


def train_models(params, test_mode=False):
    dic_var_epi_lens = [{}]
    if params.get('is_variable_max_episode_len', False):
        dic_var_epi_lens = params['par_variable_max_episode_lens']

    for scenario in params['scenarios']:
        for dic_var_epi_len in dic_var_epi_lens:
            for i, num_episode in enumerate(params['num_episodes']):
                _train_model(i, scenario, num_episode, params, dic_var_epi_len, test_mode)


if __name__ == '__main__':
    fn_param = sys.argv[1]
    try:
        test_mode = eval(sys.argv[2])
    except IndexError:
        test_mode = False

    dics_pars = eval(open(fn_param).read())
    if isinstance(dics_pars, dict):
        dics_pars = [dics_pars]
    for dic_par in dics_pars:
        train_models(dic_par, test_mode)
