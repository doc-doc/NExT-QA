# ====================================================
# @Time    : 12/2/19 9:18 AM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : utils.py
# ====================================================
import skimage.io as sio
import numpy as np
import torch
from torchvision import transforms as trn
from torch.autograd import Variable
from skimage.transform import resize
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


preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

def load_image_fn(filename, H, W):
    """
    load image and transfer to torch tensor
    :param filename:
    :return:
    """
    img = sio.imread(filename)

    img = resize(img, (H, W), mode='constant')


    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)

    img = img.astype('float32') / 255.0
    img = torch.from_numpy(img.transpose([2, 0, 1])).cuda()
    with torch.no_grad():
        img = Variable(preprocess(img))

    return img


def get_clip(clip, verbose=True):
    """
    Loads a clip to be fed to C3D for classification.
    TODO: should I remove mean here?

    Parameters
    ----------
    clip_name: str
        the name of the clip (subfolder in 'data').
    verbose: bool
        if True, shows the unrolled clip (default is True).

    Returns
    -------
    Tensor
        a pytorch batch (n, ch, fr, h, w).
    """

    # clip = sorted(glob(join('data', clip_name, '*.png')))
    clip = np.array([resize(sio.imread(frame), output_shape=(112, 200), preserve_range=True, mode='constant') for frame in clip])
    clip = clip[:, :, 44:44 + 112, :]  # crop centrally

    if verbose:
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
        sio.imshow(clip_img.astype(np.uint8))
        sio.show()

    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    # clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)

    return torch.from_numpy(clip)
