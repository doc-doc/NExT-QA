import torch
import torch.nn as nn
import random as rd
from .Attention import TempAttentionHis, TempAttention, SpatialAttention
from .memory_rand import MemoryRamTwoStreamModule, MemoryRamModule, MMModule
from .memory_module import EpisodicMemory
from .q_v_transformer import CoAttention
from .gcn import AdjLearner, GCN
from block import fusions #pytorch >= 1.1.0


class EVQA(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, device):
        """

        :param vid_encoder:
        :param qns_encoder:
        :param ans_decoder:
        :param device:
        """
        super(EVQA, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.device = device
        self.FC = nn.Linear(qns_encoder.dim_hidden, 1)

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

        vid_outputs, vid_hidden = self.vid_encoder(vid_feats)
        qns_outputs, qns_hidden = self.qns_encoder(qns, qns_lengths)


        qns_embed = qns_hidden[0].squeeze()
        vid_embed = vid_hidden[0].squeeze()

        # print(qns_embed.shape, vid_embed.shape)

        fuse = qns_embed + vid_embed

        outputs = self.FC(fuse).squeeze()

        return outputs

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
            nn.Softmax(dim=-1))

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
        local_out = torch.sum(q_v_output * local_attn, dim=1)

        # print(qns_last_hidden.shape, vid_last_hidden.shape)
        # qns_embed = qns_last_hidden.permute(1, 0, 2).contiguous().view(vid_feats.shape[0], -1)
        global_out = self.global_fusion((qns_last_hidden, v_last_hidden))


        out = self.fusion((global_out, local_out)).squeeze()

        return out

