import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'networks')
from Attention import TempAttention, SpatialAttention
from memory_rand import MemoryRamTwoStreamModule, MemoryRamModule, MMModule



class HME(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, max_len_v, max_len_q, device, input_drop_p=0.2):
        """
        Heterogeneous memory enhanced multimodal attention model for video question answering (CVPR19)
        :param vid_encoder:
        :param qns_encoder:
        :param ans_decoder:
        :param device:
        """
        super(HME, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder


        dim = qns_encoder.dim_hidden

        self.temp_att_a = TempAttention(dim * 2, dim * 2, hidden_dim=256)
        self.temp_att_m = TempAttention(dim * 2, dim * 2, hidden_dim=256)
        self.mrm_vid = MemoryRamTwoStreamModule(dim, dim, max_len_v, device)
        self.mrm_txt = MemoryRamModule(dim, dim, max_len_q, device)

        self.mm_module_v1 = MMModule(dim, input_drop_p, device)

        self.linear_vid = nn.Linear(dim*2, dim)
        self.linear_qns = nn.Linear(dim*2, dim)
        self.linear_mem = nn.Linear(dim*2, dim)
        self.vq2word_hme = nn.Linear(dim*3, 1)
        self.device = device

    def forward(self, vid_feats, qas, qas_lengths):
        """

        :param vid_feats:
        :param qns:
        :param qns_lengths:
        :param mode:
        :return:
        """
        if self.qns_encoder.use_bert:
            cand_qas = qas.permute(1, 0, 2, 3)  # for BERT
        else:
            cand_qas = qas.permute(1, 0, 2)
        cand_len = qas_lengths.permute(1, 0)

        outputs_app_l1, outputs_app_l2, outputs_motion_l1, outputs_motion_l2 = self.vid_encoder(vid_feats)
        vid_feats = (outputs_app_l1, outputs_app_l2, outputs_motion_l1, outputs_motion_l2)
        out = []
        for idx, qa in enumerate(cand_qas):
            encoder_out = self.vq_encoder(vid_feats, qa, cand_len[idx])
            out.append(encoder_out)

        out = torch.stack(out, 0).transpose(1, 0)

        _, predict_idx = torch.max(out, 1)

        return out, predict_idx

    def vq_encoder(self, vid_feats, qns, qns_lengths, iter_num=3):

        """
        :param vid_feats:
        :param qns:
        :param qns_lengths:
        :param ans:
        :param ans_lengths:
        :return:
        """

        outputs_app_l1, outputs_app_l2, outputs_motion_l1, outputs_motion_l2 = vid_feats
        outputs_app = torch.cat((outputs_app_l1, outputs_app_l2), dim=-1)
        outputs_motion = torch.cat((outputs_motion_l1, outputs_motion_l2), dim=-1)

        batch_size, fnum, vid_feat_dim = outputs_app.size()

        qns_output, qns_hidden = self.qns_encoder(qns, qns_lengths)
        # print(qns_output.shape, qns_hidden[0].shape) #torch.Size([10, 23, 256]) torch.Size([2, 10, 256])


        # qns_output = qns_output.permute(1, 0, 2)
        batch_size, seq_len, qns_feat_dim = qns_output.size()

        qns_embed = qns_hidden[0].permute(1, 0, 2).contiguous().view(batch_size, -1) #(batch_size, feat_dim)

        # Apply temporal attention
        att_app, beta_app = self.temp_att_a(qns_embed, outputs_app)
        att_motion, beta_motion = self.temp_att_m(qns_embed, outputs_motion)
        tmp_app_motion = torch.cat((outputs_app_l2[:, -1, :], outputs_motion_l2[:, -1, :]), dim=-1)

        mem_output = torch.zeros(batch_size, vid_feat_dim).to(self.device)

        for bs in range(batch_size):
            mem_ram_vid = self.mrm_vid(outputs_app_l2[bs], outputs_motion_l2[bs], fnum)
            cur_qns = qns_output[bs][:qns_lengths[bs]]
            mem_ram_txt = self.mrm_txt(cur_qns, qns_lengths[bs]) #should remove padded zeros
            mem_output[bs] = self.mm_module_v1(tmp_app_motion[bs].unsqueeze(0), mem_ram_vid, mem_ram_txt, iter_num)

        app_trans = torch.tanh(self.linear_vid(att_app))
        motion_trans = torch.tanh(self.linear_vid(att_motion))
        mem_trans = torch.tanh(self.linear_mem(mem_output))

        encoder_outputs = torch.cat((app_trans, motion_trans, mem_trans), dim=1)
        outputs = self.vq2word_hme(encoder_outputs).squeeze()

        return outputs