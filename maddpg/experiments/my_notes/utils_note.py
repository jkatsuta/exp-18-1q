import os
import sys
import glob
import os.path as osp
from IPython.display import HTML


def get_all_scenarios(exp_dir):
    exp_dirs =\
        [osp.basename(d) for d in glob.glob(osp.join(exp_dir, '*'))]
    return exp_dirs


def print_load_pars(exp_dir, par_template):
    all_scenarios = get_all_scenarios(exp_dir)
    for scenario in all_scenarios:
        print(par_template % (exp_dir, scenario))


def print_load_pars2():
    loaded_par_file = './params_play/loaded_models_exp11.dics'
    exp_dir = './exp_results/exp11_180716/'

    par_template =\
"""{
'p_dir': '%s',
'exp_dir': '%s',
'n_epis': [40000],
'num_episodes': 5,
'max_episode_len': 50,
'good_policy': 'ddpg',
'adv_policy': 'ddpg',
'outfile_suffix': None,
'display_speed': 'slow',
'exec': True,
},"""
    os.chdir('../')
    with open(loaded_par_file, 'w') as g:
        g.write('[\n')
        sys.stdout = g
        print_load_pars(exp_dir, par_template)
        sys.stdout = sys.__stdout__
        g.write(']')


def play_linked_video(each_exp_dir, fn_video, width=500, height=300):
    fn_video = _get_linked_video(each_exp_dir, fn_video)
    return _play_video(fn_video, width, height)


def _get_linked_video(each_exp_dir, fn_video):
    link_dir = osp.join('./videos', osp.basename(each_exp_dir))
    if osp.lexists(link_dir):
        os.remove(link_dir)
    os.symlink(osp.abspath(each_exp_dir), link_dir)
    fn_video = osp.join(link_dir, 'videos/%s' % fn_video)
    return fn_video


def _play_video(fn_video, width=500, height=300):
    print(fn_video)
    return HTML("""
    <video width="%d" height="%d" controls>
      <source src="%s" type="video/mp4">
    </video>""" % (width, height, fn_video))
