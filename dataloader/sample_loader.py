import torch
from torch.utils.data import Dataset, DataLoader
from .util import load_file, pkdump, pkload
import os.path as osp
import numpy as np
import nltk
import pandas as pd
import json
import string
import h5py
import pickle as pkl

class VidQADataset(Dataset):
    """load the dataset in dataloader"""

    def __init__(self, video_feature_path, video_feature_cache, sample_list_path, vocab, use_bert, mode):
        self.video_feature_path = video_feature_path
        self.vocab = vocab
        sample_list_file = osp.join(sample_list_path, '{}.csv'.format(mode))
        self.sample_list = load_file(sample_list_file)
        self.video_feature_cache = video_feature_cache
        self.max_qa_length = 37
        self.use_bert = use_bert
        self.use_frame = True
        self.use_mot = True
        if self.use_bert:
            self.bert_file = osp.join(video_feature_path, 'qas_bert/bert_ft_{}.h5'.format(mode))

        vid_feat_file = osp.join(video_feature_path, 'vid_feat/app_mot_{}.h5'.format(mode))
        print('Load {}...'.format(vid_feat_file))
        self.frame_feats = {}
        self.mot_feats = {}
        with h5py.File(vid_feat_file, 'r') as fp:
            vids = fp['ids']
            feats = fp['feat']
            for id, (vid, feat) in enumerate(zip(vids, feats)):
                if self.use_frame:
                    self.frame_feats[str(vid)] = feat[:, :2048]  # (16, 2048)
                if self.use_mot:
                    self.mot_feats[str(vid)] = feat[:, 2048:]  # (16, 2048)


    def __len__(self):
        return len(self.sample_list)


    def get_video_feature(self, video_name):
        """
        :param video_name:
        :return:
        """
        if self.use_frame:
            app_feat = self.frame_feats[video_name]
            video_feature = app_feat # (16, 2048)
        if self.use_mot:
            mot_feat = self.mot_feats[video_name]
            video_feature = np.concatenate((video_feature, mot_feat), axis=1) #(16, 4096)

        return torch.from_numpy(video_feature).type(torch.float32)


    def get_word_idx(self, text):
        """
        """
        tokens = nltk.tokenize.word_tokenize(str(text).lower())
        token_ids = [self.vocab(token) for i, token in enumerate(tokens) if i < 25]

        return token_ids

    def get_trans_matrix(self, candidates):

        qa_lengths = [len(qa) for qa in candidates]
        candidates_matrix = torch.zeros([5, self.max_qa_length]).long()
        for k in range(5):
            sentence = candidates[k]
            candidates_matrix[k, :qa_lengths[k]] = torch.Tensor(sentence)

        return candidates_matrix, qa_lengths



    def __getitem__(self, idx):
        """
        """
        cur_sample = self.sample_list.loc[idx]
        video_name, qns, ans, qid = str(cur_sample['video']), str(cur_sample['question']),\
                                    int(cur_sample['answer']), str(cur_sample['qid'])
        candidate_qas = []
        qns2ids = [self.vocab('<start>')]+self.get_word_idx(qns)+[self.vocab('<end>')]
        for id in range(5):
            cand_ans = cur_sample['a'+str(id)]
            ans2id = self.get_word_idx(cand_ans) + [self.vocab('<end>')]
            candidate_qas.append(qns2ids+ans2id)

        candidate_qas, qa_lengths = self.get_trans_matrix(candidate_qas)
        if self.use_bert:
            with h5py.File(self.bert_file, 'r') as fp:
                temp_feat = fp['feat'][idx]
                candidate_qas = torch.from_numpy(temp_feat).type(torch.float32)
            for i in range(5):
                valid_row = nozero_row(candidate_qas[i])
                qa_lengths[i] = valid_row

        video_feature = self.get_video_feature(video_name)
        qns_key = video_name + '_' + qid
        qa_lengths = torch.tensor(qa_lengths)
        

        return video_feature, candidate_qas, qa_lengths, ans, qns_key



def nozero_row(A):
    i = 0
    for row in A:
        if row.sum()==0:
            break
        i += 1

    return i



class QALoader():
    def __init__(self, batch_size, num_worker, video_feature_path, video_feature_cache,
                 sample_list_path, vocab, use_bert, train_shuffle=True, val_shuffle=False):
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.video_feature_path = video_feature_path
        self.video_feature_cache = video_feature_cache
        self.sample_list_path = sample_list_path
        self.vocab = vocab
        self.use_bert = use_bert

        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle


    def run(self, mode=''):
        if mode != 'train':
            train_loader = ''
            val_loader = self.validate(mode)
        else:
            train_loader = self.train('train')
            val_loader = self.validate('val')
        return train_loader, val_loader


    def train(self, mode):

        training_set = VidQADataset(self.video_feature_path, self.video_feature_cache, self.sample_list_path,
                                       self.vocab, self.use_bert, mode)

        print('Eligible video-qa pairs for training : {}'.format(len(training_set)))
        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_worker
            )

        return train_loader


    def validate(self, mode):

        validation_set = VidQADataset(self.video_feature_path, self.video_feature_cache, self.sample_list_path,
                                         self.vocab, self.use_bert, mode)

        print('Eligible video-qa pairs for validation : {}'.format(len(validation_set)))
        val_loader = DataLoader(
            dataset=validation_set,
            batch_size=self.batch_size,
            shuffle=self.val_shuffle,
            num_workers=self.num_worker
            )

        return val_loader

