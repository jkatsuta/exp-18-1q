#!/usr/bin/env python
import os
import sys
import re
import os.path as osp


def _get_scenario_of_oldexp(model):
    repat = re.compile('^exp_([!-~]+)_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$')
    for dir_name in model.split('/'):
        m = repat.match(dir_name)
        if m:
            scenario_name = m.group(1)
            return scenario_name


def _get_scenario_of_newexp(model):
    for dir_name in model.split('/'):
        if dir_name.find('__') > 0:
            scenario_name = dir_name.split('__')[0]
            return scenario_name


def get_scenario_name(model):
    if model.find('__') > 0:
        return _get_scenario_of_newexp(model)
    else:
        return _get_scenario_of_oldexp(model)


def _get_video_dir(model):
    video_dir = osp.dirname(model)
    if osp.dirname(video_dir):
        video_dir = osp.dirname(video_dir)
    return osp.join(video_dir, 'videos')


def get_video_file_name(model, suffix, seed):
    video_dir = _get_video_dir(model)
    if not osp.exists(video_dir):
        os.makedirs(video_dir)
    n_iter = model.split('-')[-1]

    if seed is None:
        seed = ''
    else:
        seed = '_seed%d' % seed

    if suffix is None:
        basename = 'video-%s%s.mp4' % (n_iter, seed)
    else:
        basename = 'video-%s_%s.mp4' % (n_iter, suffix+seed)
    return osp.join(video_dir, basename)


def get_display_speed(key):
    # ret value: n_per_frame, sleep_time
    if key == 'slow':
        return 10, 0.1
    elif key == 'normal':
        return 20, 0.03
    elif key == 'high':
        return 30, 0.01
    elif key == 'very-slow':
        return 3, 0.1  # applied to n_per_frame
    else:
        print('use slow/normal/high for display_speed.')
        exit(1)


def get_seeds(pars):
    seeds = pars.get('seeds', [None])
    if isinstance(seeds, int):
        seeds = [seeds]
    return seeds


def main(trained_model, pars, seed=None, test_mode=False):
    scenario = get_scenario_name(trained_model)
    video_file_name =\
        get_video_file_name(trained_model, pars.get('outfile_suffix', None), seed)

    com = 'python train.py --display --num-episodes %d ' % pars['num_episodes']
    com += '--scenario %s --load-model %s ' % (scenario, trained_model)
    com += '--max-episode-len %d ' % pars['max_episode_len']
    if 'good_policy' in pars.keys():
        com += '--good-policy %s ' % pars['good_policy']
    if 'adv_policy' in pars.keys():
        com += '--adv-policy %s ' % pars['adv_policy']
    if not pars.get('skip_video_record', False):
        com += '--video-record --video-file-name %s ' % video_file_name
    if pars['display_speed'] is not None:
        frames_per_sec, sleep_time = get_display_speed(pars['display_speed'])
        com += '--video-frames-per-second %d ' % frames_per_sec
        com += '--display-sleep-second %f ' % sleep_time
    if seed is not None:
        com += '--seed %d ' % seed
    print(com)
    if not test_mode:
        os.system(com)


if __name__ == '__main__':
    fn_par = sys.argv[1]
    try:
        test_mode = eval(sys.argv[2])
    except IndexError:
        test_mode = False

    dics_pars = eval(open(fn_par).read())
    for dic_par in dics_pars:
        seeds = get_seeds(dic_par)
        for seed in seeds:
            for n_epi in dic_par['n_epis']:
                model = osp.join(dic_par['p_dir'], dic_par['exp_dir'],
                                 'models/model-%d' % n_epi)
                main(model, dic_par, seed, test_mode)
