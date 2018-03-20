#!/usr/bin/env python
import os
import re
import argparse
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


def get_video_file_name(model, suffix):
    video_dir = _get_video_dir(model)
    if not osp.exists(video_dir):
        os.makedirs(video_dir)
    n_iter = model.split('-')[-1]
    if suffix is None:
        basename = 'video-%s.mp4' % n_iter
    else:
        basename = 'video-%s_%s.mp4' % (n_iter, suffix)
    return osp.join(video_dir, basename)


def get_display_speed(key):
    # ret value: n_per_frame, sleep_time
    if key == 'slow':
        return 10, 0.1
    elif key == 'normal':
        return 20, 0.03
    elif key == 'high':
        return 30, 0.01
    else:
        print('use slow/normal/high for display_speed.')
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Play trained model of MADDPG')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--num-episodes', type=int, default=5, help='number of episodes')
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    parser.add_argument('--skip-video-record', action='store_true', default=False)
    parser.add_argument('--outfile-suffix', type=str, default=None)
    parser.add_argument('--display-speed', type=str, default=None, help='slow/normal/high')
    args = parser.parse_args()

    trained_model = args.model
    scenario = get_scenario_name(trained_model)
    video_file_name = get_video_file_name(trained_model, args.outfile_suffix)

    com = 'python train.py --display --num-episodes %d ' % args.num_episodes
    com += '--scenario %s --load-model %s ' % (scenario, trained_model)
    com += '--max-episode-len %d ' % args.max_episode_len
    com += '--good-policy %s --adv-policy %s ' % (args.good_policy, args.adv_policy)
    if not args.skip_video_record:
        com += '--video-record --video-file-name %s ' % video_file_name
    if args.display_speed is not None:
        frames_per_sec, sleep_time = get_display_speed(args.display_speed)
        com += '--video-frames-per-second %d ' % frames_per_sec
        com += '--display-sleep-second %f ' % sleep_time
    os.system(com)
    # print(com)
