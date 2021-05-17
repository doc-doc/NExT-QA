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


