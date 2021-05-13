# ====================================================
# @Time    : 11/14/19 10:53 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : util.py
# ====================================================
import json
import os
import os.path as osp
import numpy as np
import pickle as pkl
import pandas as pd

def load_file(file_name):
    annos = None
    if osp.splitext(file_name)[-1] == '.csv':
        return pd.read_csv(file_name)
    with open(file_name, 'r') as fp:
        if osp.splitext(file_name)[1]== '.txt':
            annos = fp.readlines()
            annos = [line.rstrip() for line in annos]
        if osp.splitext(file_name)[1] == '.json':
            annos = json.load(fp)

    return annos

def save_file(obj, filename):
    """
    save obj to filename
    :param obj:
    :param filename:
    :return:
    """
    filepath = osp.dirname(filename)
    if filepath != '' and not osp.exists(filepath):
        os.makedirs(filepath)
    else:
        with open(filename, 'w') as fp:
            json.dump(obj, fp, indent=4)

def pkload(file):
    data = None
    if osp.exists(file) and osp.getsize(file) > 0:
        with open(file, 'rb') as fp:
            data = pkl.load(fp)
        # print('{} does not exist'.format(file))
    return data


def pkdump(data, file):
    dirname = osp.dirname(file)
    if not osp.exists(dirname):
        os.makedirs(dirname)
    with open(file, 'wb') as fp:
        pkl.dump(data, fp)


def get_video_frames(video_relation_file):

    folders = load_file(video_relation_file)
    vframes = {}
    for recode in folders:
        video, nframe = recode[0], recode[1]
        if video not in vframes:
            vframes[video] = nframe
        else:
            continue

    all_frames = []
    sample_num = 512
    # miss_videos = load_file('dataset/vidor/miss_videos.json')
    print(len(vframes))

    for video, nframe in vframes.items():
        #if video not in miss_videos: continue
        #if video not in ['1021/3726334221', '0003/6855479096']: continue
        samples = np.round(np.linspace(
            1, nframe, sample_num))

        samples = set([int(s) for s in samples])
        samples = list(samples)
        fnames = [osp.join(video, str(fid).zfill(6)) for fid in samples]
        if all_frames == []:
            all_frames = fnames
        else:
            all_frames.extend(fnames)

    print(len(all_frames))
    return all_frames

