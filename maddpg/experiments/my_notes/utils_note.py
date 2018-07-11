import os
import os.path as osp
from IPython.display import HTML


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
