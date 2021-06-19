import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'networks')
from q_v_transformer import CoAttention
from gcn import AdjLearner, GCN
from block import fusions #pytorch >= 1.1.0


class HGA(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, device):
        """
        Reasoning with Heterogeneous Graph Alignment for Video Question Answering (AAAI2020)
        :param vid_encoder:
        :param qns_encoder:
        :param device:
        """
        super(HGA, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.device = device
        hidden_size = vid_encoder.dim_hidden
        input_dropout_p = vid_encoder.input_dropout_p

        self.q_input_ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.v_input_ln = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.co_attn = CoAttention(
            hidden_size, n_layers=vid_encoder.n_layers, dropout_p=input_dropout_p)

        self.adj_learner = AdjLearner(
            hidden_size, hidden_size, dropout=input_dropout_p)

        self.gcn = GCN(
            hidden_size,
            hidden_size,
            hidden_size,
            num_layers=2,
            dropout=input_dropout_p)

        self.gcn_atten_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=-1)) #change to dim=-2 for attention-pooling otherwise sum-pooling

        self.global_fusion = fusions.Block(
            [hidden_size, hidden_size], hidden_size, dropout_input=input_dropout_p)

        self.fusion = fusions.Block([hidden_size, hidden_size], 1)


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

        v_output, v_hidden = self.vid_encoder(vid_feats)
        v_last_hidden = torch.squeeze(v_hidden)

        out = []
        for idx, qa in enumerate(cand_qas):
            encoder_out = self.vq_encoder(v_output, v_last_hidden, qa, cand_len[idx])
            out.append(encoder_out)

        out = torch.stack(out, 0).transpose(1, 0)
        _, predict_idx = torch.max(out, 1)


        return out, predict_idx


    def vq_encoder(self, v_output, v_last_hidden, qas, qas_lengths):
        """
        :param vid_feats:
        :param qas:
        :param qas_lengths:
        :return:
        """
        q_output, s_hidden = self.qns_encoder(qas, qas_lengths)
        qns_last_hidden = torch.squeeze(s_hidden)

        q_output = self.q_input_ln(q_output)
        v_output = self.v_input_ln(v_output)

        q_output, v_output = self.co_attn(q_output, v_output)

        ### GCN
        adj = self.adj_learner(q_output, v_output)
        # q_v_inputs of shape (batch_size, q_v_len, hidden_size)
        q_v_inputs = torch.cat((q_output, v_output), dim=1)
        # q_v_output of shape (batch_size, q_v_len, hidden_size)
        q_v_output = self.gcn(q_v_inputs, adj)

        ## attention pool
        local_attn = self.gcn_atten_pool(q_v_output)
        print(local_attn)
        local_out = torch.sum(q_v_output * local_attn, dim=1)

        # print(qns_last_hidden.shape, vid_last_hidden.shape)
        # qns_embed = qns_last_hidden.permute(1, 0, 2).contiguous().view(vid_feats.shape[0], -1)
        global_out = self.global_fusion((qns_last_hidden, v_last_hidden))


        out = self.fusion((global_out, local_out)).squeeze()

        return out
