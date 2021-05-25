import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'networks')
from memory_module import EpisodicMemory


class CoMem(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, max_len_v, max_len_q, device):
        """
        motion-appearance co-memory networks for video question answering (CVPR18)
        :param vid_encoder:
        :param qns_encoder:
        :param ans_decoder:
        :param device:
        """
        super(CoMem, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder

        dim = qns_encoder.dim_hidden

        self.epm_app = EpisodicMemory(dim*2)
        self.epm_mot = EpisodicMemory(dim*2)

        self.linear_ma = nn.Linear(dim*2*3, dim*2)
        self.linear_mb = nn.Linear(dim*2*3, dim*2)

        self.vq2word = nn.Linear(dim*2*2, 1)

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
        Co-memory network
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

        qns_output, qns_hidden = self.qns_encoder(qns, qns_lengths)

        # qns_output = qns_output.permute(1, 0, 2)
        batch_size, seq_len, qns_feat_dim = qns_output.size()

        qns_embed = qns_hidden[0].permute(1, 0, 2).contiguous().view(batch_size, -1) #(batch_size, feat_dim)

        m_app = outputs_app[:, -1, :]
        m_mot = outputs_motion[:, -1, :]
        ma, mb = m_app.detach(), m_mot.detach()
        m_app = m_app.unsqueeze(1)
        m_mot = m_mot.unsqueeze(1)
        for _ in range(iter_num):
            mm = ma + mb
            m_app = self.epm_app(outputs_app, mm, m_app)
            m_mot = self.epm_mot(outputs_motion, mm, m_mot)
            ma_q = torch.cat((ma, m_app.squeeze(1), qns_embed), dim=1)
            mb_q = torch.cat((mb, m_mot.squeeze(1), qns_embed), dim=1)
            ma = torch.tanh(self.linear_ma(ma_q))
            mb = torch.tanh(self.linear_mb(mb_q))

        mem = torch.cat((ma, mb), dim=1)
        outputs = self.vq2word(mem).squeeze()

        return outputs