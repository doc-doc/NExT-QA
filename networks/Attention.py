import torch
import torch.nn as nn
import torch.nn.functional as F


class TempAttention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, text_dim, visual_dim, hidden_dim):
        super(TempAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear_text = nn.Linear(text_dim, hidden_dim)
        self.linear_visual = nn.Linear(visual_dim, hidden_dim)
        self.linear_att = nn.Linear(hidden_dim, 1, bias=False)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_normal_(self.linear_text.weight)
        nn.init.xavier_normal_(self.linear_visual.weight)
        nn.init.xavier_normal_(self.linear_att.weight)

    def forward(self, qns_embed, vid_outputs):
        """
        Arguments:
            qns_embed {Variable} -- batch_size x dim
            vid_outputs {Variable} -- batch_size x seq_len x dim

        Returns:
            context -- context vector of size batch_size x dim
        """
        qns_embed_trans = self.linear_text(qns_embed)

        batch_size, seq_len, visual_dim = vid_outputs.size()
        vid_outputs_temp = vid_outputs.contiguous().view(batch_size*seq_len, visual_dim)
        vid_outputs_trans = self.linear_visual(vid_outputs_temp)
        vid_outputs_trans = vid_outputs_trans.view(batch_size, seq_len, self.hidden_dim)

        qns_embed_trans = qns_embed_trans.unsqueeze(1).repeat(1, seq_len, 1)


        o = self.linear_att(torch.tanh(qns_embed_trans+vid_outputs_trans))

        e = o.view(batch_size, seq_len)
        beta = F.softmax(e, dim=1)
        context = torch.bmm(beta.unsqueeze(1), vid_outputs).squeeze(1)

        return context, beta


class SpatialAttention(nn.Module):
    """
    Apply spatial attention on vid feature before being fed into LSTM
    """

    def __init__(self, text_dim=1024, vid_dim=3072, hidden_dim=512, input_dropout_p=0.2):
        super(SpatialAttention, self).__init__()

        self.linear_v = nn.Linear(vid_dim, hidden_dim)
        self.linear_q = nn.Linear(text_dim, hidden_dim)
        self.linear_att = nn.Linear(hidden_dim, 1, bias=False)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(input_dropout_p)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_normal_(self.linear_v.weight)
        nn.init.xavier_normal_(self.linear_q.weight)
        nn.init.xavier_normal_(self.linear_att.weight)

    def forward(self, qns_feat, vid_feats):
        """
        Apply question feature as semantic clue to guide feature aggregation at each frame
        :param vid_feats: fnum x feat_dim x 7 x 7
        :param qns_feat: dim_hidden*2
        :return:
        """
        # print(qns_feat.size(), vid_feats.size())
        # permute to fnum x 7 x 7 x feat_dim
        vid_feats = vid_feats.permute(0, 2, 3, 1)
        fnum, width, height, feat_dim = vid_feats.size()
        vid_feats = vid_feats.contiguous().view(-1, feat_dim)
        vid_feats_trans = self.linear_v(vid_feats)

        vid_feats_trans = vid_feats_trans.view(fnum, width*height, -1)
        region_num = vid_feats_trans.shape[1]

        qns_feat_trans = self.linear_q(qns_feat)

        qns_feat_trans = qns_feat_trans.repeat(fnum, region_num, 1)
        # print(vid_feats_trans.shape, qns_feat_trans.shape)

        vid_qns = self.linear_att(torch.tanh(vid_feats_trans + qns_feat_trans))

        vid_qns_o = vid_qns.view(fnum, region_num)
        alpha = self.softmax(vid_qns_o)
        alpha = alpha.unsqueeze(1)
        vid_feats = vid_feats.view(fnum, region_num, -1)
        feature = torch.bmm(alpha, vid_feats).squeeze(1)
        feature = self.dropout(feature)
        # print(feature.size())
        return feature, alpha


class TempAttentionHis(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, visual_dim, text_dim, his_dim, mem_dim):
        super(TempAttentionHis, self).__init__()
        # self.dim = dim
        self.mem_dim = mem_dim
        self.linear_v = nn.Linear(visual_dim, self.mem_dim, bias=False)
        self.linear_q = nn.Linear(text_dim, self.mem_dim, bias=False)
        self.linear_his1 = nn.Linear(his_dim, self.mem_dim, bias=False)
        self.linear_his2 = nn.Linear(his_dim, self.mem_dim, bias=False)
        self.linear_att = nn.Linear(self.mem_dim, 1, bias=False)
        self._init_weight()


    def _init_weight(self):
        nn.init.xavier_normal_(self.linear_v.weight)
        nn.init.xavier_normal_(self.linear_q.weight)
        nn.init.xavier_normal_(self.linear_his1.weight)
        nn.init.xavier_normal_(self.linear_his2.weight)
        nn.init.xavier_normal_(self.linear_att.weight)


    def forward(self, qns_embed, vid_outputs, his):
        """
        :param qns_embed: batch_size x 1024
        :param vid_outputs: batch_size x seq_num x feat_dim
        :param his: batch_size x 512
        :return:
        """

        batch_size, seq_len, feat_dim = vid_outputs.size()
        vid_outputs_trans = self.linear_v(vid_outputs.contiguous().view(batch_size * seq_len, feat_dim))
        vid_outputs_trans = vid_outputs_trans.view(batch_size, seq_len, self.mem_dim)

        qns_embed_trans = self.linear_q(qns_embed)
        qns_embed_trans = qns_embed_trans.unsqueeze(1).repeat(1, seq_len, 1)


        his_trans = self.linear_his1(his)
        his_trans = his_trans.unsqueeze(1).repeat(1, seq_len, 1)

        o = self.linear_att(torch.tanh(qns_embed_trans + vid_outputs_trans + his_trans))

        e = o.view(batch_size, seq_len)
        beta = F.softmax(e, dim=1)
        context = torch.bmm(beta.unsqueeze(1), vid_outputs_trans).squeeze(1)

        his_acc = torch.tanh(self.linear_his2(his))

        context += his_acc

        return context, beta


class MultiModalAttentionModule(nn.Module):

    def __init__(self, hidden_size=512, simple=False):
        """Set the hyper-parameters and build the layers."""
        super(MultiModalAttentionModule, self).__init__()

        self.hidden_size = hidden_size
        self.simple = simple

        # alignment model
        # see appendices A.1.2 of neural machine translation

        self.Wav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        self.Wat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        self.Uav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        self.Uat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        self.Vav = nn.Parameter(torch.FloatTensor(hidden_size), requires_grad=True)
        self.Vat = nn.Parameter(torch.FloatTensor(hidden_size), requires_grad=True)
        self.bav = nn.Parameter(torch.FloatTensor(1, 1, hidden_size), requires_grad=True)
        self.bat = nn.Parameter(torch.FloatTensor(1, 1, hidden_size), requires_grad=True)

        self.Whh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        self.Wvh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        self.Wth = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        self.bh = nn.Parameter(torch.FloatTensor(1, 1, hidden_size), requires_grad=True)

        self.video_sum_encoder = nn.Linear(hidden_size, hidden_size)
        self.question_sum_encoder = nn.Linear(hidden_size, hidden_size)

        self.Wb = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        self.Vbv = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        self.Vbt = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        self.bbv = nn.Parameter(torch.FloatTensor(hidden_size), requires_grad=True)
        self.bbt = nn.Parameter(torch.FloatTensor(hidden_size), requires_grad=True)
        self.wb = nn.Parameter(torch.FloatTensor(hidden_size), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        self.Wav.data.normal_(0.0, 0.1)
        self.Wat.data.normal_(0.0, 0.1)
        self.Uav.data.normal_(0.0, 0.1)
        self.Uat.data.normal_(0.0, 0.1)
        self.Vav.data.normal_(0.0, 0.1)
        self.Vat.data.normal_(0.0, 0.1)
        self.bav.data.fill_(0)
        self.bat.data.fill_(0)

        self.Whh.data.normal_(0.0, 0.1)
        self.Wvh.data.normal_(0.0, 0.1)
        self.Wth.data.normal_(0.0, 0.1)
        self.bh.data.fill_(0)

        self.Wb.data.normal_(0.0, 0.01)
        self.Vbv.data.normal_(0.0, 0.01)
        self.Vbt.data.normal_(0.0, 0.01)
        self.wb.data.normal_(0.0, 0.01)

        self.bbv.data.fill_(0)
        self.bbt.data.fill_(0)

    def forward(self, h, hidden_frames, hidden_text, inv_attention=False):
        # print self.Uav
        # hidden_text:  1 x T1 x 1024 (looks like a two layer one-directional LSTM, combining each layer's hidden)
        # hidden_frame: 1 x T2 x 1024 (from video encoder output, 1024 is similar from above)

        # print hidden_frames.size(),hidden_text.size()
        Uhv = torch.matmul(h, self.Uav)  # (1,512)
        Uhv = Uhv.view(Uhv.size(0), 1, Uhv.size(1))  # (1,1,512)

        Uht = torch.matmul(h, self.Uat)  # (1,512)
        Uht = Uht.view(Uht.size(0), 1, Uht.size(1))  # (1,1,512)

        # print Uhv.size(),Uht.size()

        Wsv = torch.matmul(hidden_frames, self.Wav)  # (1,T,512)
        # print Wsv.size()
        att_vec_v = torch.matmul(torch.tanh(Wsv + Uhv + self.bav), self.Vav)

        Wst = torch.matmul(hidden_text, self.Wat)  # (1,T,512)
        att_vec_t = torch.matmul(torch.tanh(Wst + Uht + self.bat), self.Vat)

        if inv_attention == True:
            att_vec_v = -att_vec_v
            att_vec_t = -att_vec_t

        att_vec_v = torch.softmax(att_vec_v, dim=1)
        att_vec_t = torch.softmax(att_vec_t, dim=1)

        att_vec_v = att_vec_v.view(att_vec_v.size(0), att_vec_v.size(1), 1)  # expand att_vec from 1xT to 1xTx1
        att_vec_t = att_vec_t.view(att_vec_t.size(0), att_vec_t.size(1), 1)  # expand att_vec from 1xT to 1xTx1

        hv_weighted = att_vec_v * hidden_frames
        hv_sum = torch.sum(hv_weighted, dim=1)
        hv_sum2 = self.video_sum_encoder(hv_sum)

        ht_weighted = att_vec_t * hidden_text
        ht_sum = torch.sum(ht_weighted, dim=1)
        ht_sum2 = self.question_sum_encoder(ht_sum)

        Wbs = torch.matmul(h, self.Wb)
        mt1 = torch.matmul(ht_sum, self.Vbt) + self.bbt + Wbs
        mv1 = torch.matmul(hv_sum, self.Vbv) + self.bbv + Wbs
        mtv = torch.tanh(torch.cat([mv1, mt1], dim=0))
        mtv2 = torch.matmul(mtv, self.wb)
        beta = torch.softmax(mtv2, dim=0)

        output = torch.tanh(torch.matmul(h, self.Whh) + beta[0] * hv_sum2 +
                            beta[1] * ht_sum2 + self.bh)
        output = output.view(output.size(1), output.size(2))

        return output