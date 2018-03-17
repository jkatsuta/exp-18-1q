#!/usr/bin/env python
import os
import sys


def _train_model(i_epi, scenario, num_episode, params, dic_var_epi_len):
    com = 'python train.py --scenario %s ' % scenario
    com += '--num-episodes %d ' % num_episode
    if len(dic_var_epi_len) > 0:
        com += '--dic-variable-max-episode-len "%s" ' % str(dic_var_epi_len)
    else:
        com += '--max-episode-len %s ' % params['max_episode_lens'][i_epi]
    if params['is_parallel']:
        com += ' &'
    os.system(com)
    # print(com)


def train_models(params):
    dic_var_epi_lens = [{}]
    if params.get('is_variable_max_episode_len', False):
        dic_var_epi_lens = params['par_variable_max_episode_lens']

    for scenario in params['scenarios']:
        for dic_var_epi_len in dic_var_epi_lens:
            for i, num_episode in enumerate(params['num_episodes']):
                _train_model(i, scenario, num_episode, params, dic_var_epi_len)


if __name__ == '__main__':
    fn_param = sys.argv[1]
    train_models(eval(open(fn_param).read()))
