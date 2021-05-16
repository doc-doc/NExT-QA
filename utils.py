import json
import os
import os.path as osp
import pickle as pkl
import pandas as pd

def set_gpu_devices(gpu_id):
    gpu = ''
    if gpu_id != -1:
        gpu = str(gpu_id)
    os.environ['CUDA_VOSIBLE_DEVICES'] = gpu


def load_file(filename):
    """
    load obj from filename
    :param filename:
    :return:
    """
    cont = None
    if not osp.exists(filename):
        print('{} not exist'.format(filename))
        return cont
    if osp.splitext(filename)[-1] == '.csv':
        # return pd.read_csv(filename, delimiter= '\t', index_col=0)
        return pd.read_csv(filename, delimiter=',')
    with open(filename, 'r') as fp:
        if osp.splitext(filename)[1] == '.txt':
            cont = fp.readlines()
            cont = [c.rstrip('\n') for c in cont]
        elif osp.splitext(filename)[1] == '.json':
            cont = json.load(fp)
    return cont


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
