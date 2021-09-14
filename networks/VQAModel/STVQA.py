import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'networks')
from Attention import TempAttention, SpatialAttention

class STVQA(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, att_dim, device):
        """
        TGIF-QA: Toward Spatio-Temporal Reasoning in Visual Question Answering
        :param vid_encoder:
        :param qns_encoder:
        :param att_dim:
        :param device:
        """
        super(STVQA, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.att_dim = att_dim

        self.spatial_att = SpatialAttention(qns_encoder.dim_hidden*2, vid_encoder.input_dim, hidden_dim=self.att_dim)
        self.temp_att = TempAttention(qns_encoder.dim_hidden*2, vid_encoder.dim_hidden*2, hidden_dim=self.att_dim)
        self.device = device

        self.FC = nn.Linear(qns_encoder.dim_hidden*2, 1)


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
        out = []
        for idx, qa in enumerate(cand_qas):
            encoder_out = self.vq_encoder(vid_feats, qa, cand_len[idx])
            out.append(encoder_out)

        out = torch.stack(out, 0).transpose(1, 0)
        _, predict_idx = torch.max(out, 1)

        return out, predict_idx

    def vq_encoder(self, vid_feats, qns, qns_lengths):
        """
        TGIF-QA: Spatial and temporal attention
        :param vid_feats: (batch_size, fnum, feat_dim, w, h)
        :param qns:
        :param qns_lengths:
        :param ans:
        :param ans_lengths:
        :param teacher_force_ratio:
        :return:
        """
        qns_output_1, qns_hidden_1 = self.qns_encoder(qns, qns_lengths)
        n_layers, batch_size, qns_dim = qns_hidden_1[0].size()

        # Concatenate the dual-layer hidden as qns embedding
        qns_embed = qns_hidden_1[0].permute(1, 0, 2) # batch first
        qns_embed = qns_embed.reshape(batch_size, -1) #(batch_size, feat_dim*2)
        batch_size, fnum, vid_dim, w, h = vid_feats.size()

        # Apply spatial attention
        vid_att_feats = torch.zeros(batch_size, fnum, vid_dim).to(self.device)
        for bs in range(batch_size):
            vid_att_feats[bs], alpha = self.spatial_att(qns_embed[bs], vid_feats[bs])

        vid_outputs, vid_hidden = self.vid_encoder(vid_att_feats)

        qns_output, qns_hidden = self.qns_encoder(qns, qns_lengths, vid_hidden)

        qns_embed = qns_hidden[0].permute(1, 0, 2).contiguous().view(batch_size, -1) #(batch_size, feat_dim)

        # Apply temporal attention
        temp_att_outputs, beta = self.temp_att(qns_embed, vid_outputs)
        encoder_outputs = (qns_embed + temp_att_outputs)
        outputs = self.FC(encoder_outputs).squeeze()

        return outputs