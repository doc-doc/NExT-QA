import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import torchnlp_nn as nlpnn


def padding_mask(seq_q, seq_k):
    # seq_k of shape (batch, k_len) and seq_q (batch, q_len), not embedded. q and k are padded with 0.
    seq_q = torch.unsqueeze(seq_q, 2)
    seq_k = torch.unsqueeze(seq_k, 2)
    pad_mask = torch.bmm(seq_q, seq_k.transpose(1, 2))
    pad_mask = pad_mask.eq(0)
    return pad_mask


def padding_mask_transformer(seq_q, seq_k):
    # original padding_mask in transformer, for masking out the padding part of key sequence.
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(
        -1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


def padding_mask_embedded(seq_q, seq_k):
    # seq_k of shape (batch, k_len, k_feat) and seq_q (batch, q_len, q_feat). q and k are padded with 0. pad_mask is (batch, q_len, k_len)
    pad_mask = torch.bmm(seq_q, seq_k.transpose(1, 2))
    pad_mask = pad_mask.eq(0)
    return pad_mask


def padding_mask_k(seq_q, seq_k):
    """ seq_k of shape (batch, k_len, k_feat) and seq_q (batch, q_len, q_feat). q and k are padded with 0. pad_mask is (batch, q_len, k_len).
    In batch 0:
    [[x x x 0]     [[0 0 0 1]
     [x x x 0]->    [0 0 0 1]
     [x x x 0]]     [0 0 0 1]] uint8
    """
    fake_q = torch.ones_like(seq_q)
    pad_mask = torch.bmm(fake_q, seq_k.transpose(1, 2))
    pad_mask = pad_mask.eq(0)
    # pad_mask = pad_mask.lt(1e-3)
    return pad_mask


def padding_mask_q(seq_q, seq_k):
    """ seq_k of shape (batch, k_len, k_feat) and seq_q (batch, q_len, q_feat). q and k are padded with 0. pad_mask is (batch, q_len, k_len).
    In batch 0:
    [[x x x x]      [[0 0 0 0]
     [x x x x]  ->   [0 0 0 0]
     [0 0 0 0]]      [1 1 1 1]] uint8
    """
    fake_k = torch.ones_like(seq_k)
    pad_mask = torch.bmm(seq_q, fake_k.transpose(1, 2))
    pad_mask = pad_mask.eq(0)
    # pad_mask = pad_mask.lt(1e-3)
    return pad_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.max_seq_len = max_seq_len

        position_encoding = np.array(
            [
                [
                    pos / np.power(10000, 2.0 * (j // 2) / d_model)
                    for j in range(d_model)
                ]
                for pos in range(max_seq_len)
            ])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat(
            (pad_row, torch.from_numpy(position_encoding).float()))

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(
            position_encoding, requires_grad=False)

    def forward(self, input_len):
        # max_len = torch.max(input_len)
        max_len = self.max_seq_len
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        input_pos = [
            list(range(1, l + 1)) + [0] * (max_len - l.item())
            for l in input_len
        ]
        input_pos = tensor(input_pos)
        return self.position_encoding(input_pos)


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=512, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        # x of shape (bs, seq_len, hs)
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class MaskedPositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim, elementwise_affine=False)

    def forward(self, x):
        # x of shape (bs, seq_len, hs)
        output = self.w2(F.relu(self.w1(x)))
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        Args:
            q: [B, L_q, D_q]
            k: [B, L_k, D_k]
            v: [B, L_v, D_v]
        """
        attention = torch.matmul(q, k.transpose(1, 2))
        if scale is not None:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        output = torch.matmul(attention, v)
        return output, attention


class MaskedScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale=None, attn_mask=None, softmax_mask=None):
        """
        Args:
            q: [B, L_q, D_q]
            k: [B, L_k, D_k]
            v: [B, L_v, D_v]
        """
        attention = torch.matmul(q, k.transpose(-2, -1))
        if scale is not None:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = attention.masked_fill(softmax_mask, 0.)
        attention = self.dropout(attention)
        output = torch.matmul(attention, v)
        return output, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(
            model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(
            model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(
            model_dim, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads)**-0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class MaskedMultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super().__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(
            model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(
            model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(
            model_dim, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = MaskedScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim, elementwise_affine=False)

    def forward(self, query, key, value, attn_mask=None, softmax_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads,
                           dim_per_head).transpose(1, 2)
        query = query.view(batch_size, -1, num_heads,
                           dim_per_head).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
        if softmax_mask is not None:
            softmax_mask = softmax_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
        # scaled dot product attention
        # key.size(-1) is 64?
        scale = key.size(-1)**-0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask, softmax_mask)

        # concat heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class SelfTransformerLayer(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super().__init__()

        self.transformer = MaskedMultiHeadAttention(
            model_dim, num_heads, dropout)
        self.feed_forward = MaskedPositionalWiseFeedForward(
            model_dim, ffn_dim, dropout)

    def forward(self, input, attn_mask=None, sf_mask=None):
        output, attention = self.transformer(
            input, input, input, attn_mask, sf_mask)
        # feed forward network
        output = self.feed_forward(output)

        return output, attention


class SelfTransformer(nn.Module):

    def __init__(
            self,
            max_len=35,
            num_layers=2,
            model_dim=512,
            num_heads=8,
            ffn_dim=2048,
            dropout=0.0,
            position=False):
        super().__init__()

        self.position = position

        self.encoder_layers = nn.ModuleList(
            [
                SelfTransformerLayer(model_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ])

        # max_seq_len is 35 or 80
        self.pos_embedding = PositionalEncoding(model_dim, max_len)

    def forward(self, input, input_length):
        # q_length of shape (batch, ), each item is the length of the seq
        if self.position:
            input += self.pos_embedding(input_length)[:, :input.size()[1], :]

        attention_mask = padding_mask_k(input, input)
        softmax_mask = padding_mask_q(input, input)

        attentions = []
        for encoder in self.encoder_layers:
            input, attention = encoder(input, attention_mask, softmax_mask)
            attentions.append(attention)

        return input, attentions


class SelfAttentionLayer(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)

        self.linear_k = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)
        self.linear_q = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)
        self.linear_v = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)

        self.linear_final = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)

        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, q, k, v, scale=None, attn_mask=None, softmax_mask=None):
        """
        Args:
            q: [B, L_q, D_q]
            k: [B, L_k, D_k]
            v: [B, L_v, D_v]
        """
        residual = q

        if attn_mask is None or softmax_mask is None:
            attn_mask = padding_mask_k(q, k)
            softmax_mask = padding_mask_q(q, k)

        # linear projection
        k = self.linear_k(k)
        v = self.linear_v(v)
        q = self.linear_q(q)

        scale = k.size(-1)**-0.5

        attention = torch.bmm(q, k.transpose(1, 2))
        if scale is not None:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = attention.masked_fill(softmax_mask, 0.)

        # attention = self.dropout(attention)
        output = torch.bmm(attention, v)
        output = self.linear_final(output)
        output = self.layer_norm(output + residual)
        return output, attention


class SelfAttention(nn.Module):

    def __init__(self, hidden_size, n_layers=1, dropout_p=0.0):
        super().__init__()

        self.encoder_layers = nn.ModuleList(
            [
                SelfAttentionLayer(hidden_size, dropout_p)
                for _ in range(n_layers)
            ])

    def forward(self, input):

        # q_attention_mask of shape (bs, q_len, v_len)
        attn_mask = padding_mask_k(input, input)
        # v_attention_mask of shape (bs, v_len, q_len)
        softmax_mask = padding_mask_q(input, input)

        attentions = []
        for encoder in self.encoder_layers:
            input, attention = encoder(
                input,
                input,
                input,
                attn_mask=attn_mask,
                softmax_mask=softmax_mask)
            attentions.append(attention)

        return input, attentions


class CoAttentionLayer(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)

        self.linear_question = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)
        self.linear_video = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)
        self.linear_v_question = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)
        self.linear_v_video = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)

        self.linear_final_qv = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)
        self.linear_final_vq = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)

        self.layer_norm_qv = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.layer_norm_vq = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(
            self,
            question,
            video,
            scale=None,
            attn_mask=None,
            softmax_mask=None,
            attn_mask_=None,
            softmax_mask_=None):
        """
        Args:
            q: [B, L_q, D_q]
            k: [B, L_k, D_k]
            v: [B, L_v, D_v]
        """
        q = question
        v = video

        if attn_mask is None or softmax_mask is None:
            attn_mask = padding_mask_k(question, video)
            softmax_mask = padding_mask_q(question, video)
        if attn_mask_ is None or softmax_mask_ is None:
            attn_mask_ = padding_mask_k(video, question)
            softmax_mask_ = padding_mask_q(video, question)

        # linear projection
        question_q = self.linear_question(question)
        video_k = self.linear_video(video)
        question = self.linear_v_question(question)
        video = self.linear_v_video(video)

        scale = video.size(-1)**-0.5

        attention_qv = torch.bmm(question_q, video_k.transpose(1, 2))
        if scale is not None:
            attention_qv = attention_qv * scale
        if attn_mask is not None:
            attention_qv = attention_qv.masked_fill(attn_mask, -np.inf)
        attention_qv = self.softmax(attention_qv)
        attention_qv = attention_qv.masked_fill(softmax_mask, 0.)

        attention_vq = torch.bmm(video_k, question_q.transpose(1, 2))
        if scale is not None:
            attention_vq = attention_vq * scale
        if attn_mask_ is not None:
            attention_vq = attention_vq.masked_fill(attn_mask_, -np.inf)
        attention_vq = self.softmax(attention_vq)
        attention_vq = attention_vq.masked_fill(softmax_mask_, 0.)

        # attention = self.dropout(attention)
        output_qv = torch.bmm(attention_qv, video)
        output_qv = self.linear_final_qv(output_qv)
        output_q = self.layer_norm_qv(output_qv + q)

        output_vq = torch.bmm(attention_vq, question)
        output_vq = self.linear_final_vq(output_vq)
        output_v = self.layer_norm_vq(output_vq + v)
        return output_q, output_v


class CoAttention(nn.Module):

    def __init__(self, hidden_size, n_layers=1, dropout_p=0.0):
        super().__init__()

        self.encoder_layers = nn.ModuleList(
            [CoAttentionLayer(hidden_size, dropout_p) for _ in range(n_layers)])

    def forward(self, question, video):
        attn_mask = padding_mask_k(question, video)
        softmax_mask = padding_mask_q(question, video)
        attn_mask_ = padding_mask_k(video, question)
        softmax_mask_ = padding_mask_q(video, question)

        for encoder in self.encoder_layers:
            question, video = encoder(
                question,
                video,
                attn_mask=attn_mask,
                softmax_mask=softmax_mask,
                attn_mask_=attn_mask_,
                softmax_mask_=softmax_mask_)

        return question, video


class CoConcatAttentionLayer(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)

        self.linear_question = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)
        self.linear_video = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)
        self.linear_v_question = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)
        self.linear_v_video = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)

        self.linear_final_qv = nn.Sequential(
            nlpnn.WeightDropLinear(
                2 * hidden_size,
                hidden_size,
                weight_dropout=dropout_p,
                bias=False), nn.ReLU(),
            nlpnn.WeightDropLinear(
                hidden_size, hidden_size, weight_dropout=dropout_p, bias=False))
        self.linear_final_vq = nn.Sequential(
            nlpnn.WeightDropLinear(
                2 * hidden_size,
                hidden_size,
                weight_dropout=dropout_p,
                bias=False), nn.ReLU(),
            nlpnn.WeightDropLinear(
                hidden_size, hidden_size, weight_dropout=dropout_p, bias=False))

        self.layer_norm_qv = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.layer_norm_vq = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(
            self,
            question,
            video,
            scale=None,
            attn_mask=None,
            softmax_mask=None,
            attn_mask_=None,
            softmax_mask_=None):
        """
        Args:
            q: [B, L_q, D_q]
            k: [B, L_k, D_k]
            v: [B, L_v, D_v]
        """
        q = question
        v = video

        if attn_mask is None or softmax_mask is None:
            attn_mask = padding_mask_k(question, video)
            softmax_mask = padding_mask_q(question, video)
        if attn_mask_ is None or softmax_mask_ is None:
            attn_mask_ = padding_mask_k(video, question)
            softmax_mask_ = padding_mask_q(video, question)

        # linear projection
        question_q = self.linear_question(question)
        video_k = self.linear_video(video)
        question = self.linear_v_question(question)
        video = self.linear_v_video(video)

        scale = video.size(-1)**-0.5

        attention_qv = torch.bmm(question_q, video_k.transpose(1, 2))
        if scale is not None:
            attention_qv = attention_qv * scale
        if attn_mask is not None:
            attention_qv = attention_qv.masked_fill(attn_mask, -np.inf)
        attention_qv = self.softmax(attention_qv)
        attention_qv = attention_qv.masked_fill(softmax_mask, 0.)

        attention_vq = torch.bmm(video_k, question_q.transpose(1, 2))
        if scale is not None:
            attention_vq = attention_vq * scale
        if attn_mask_ is not None:
            attention_vq = attention_vq.masked_fill(attn_mask_, -np.inf)
        attention_vq = self.softmax(attention_vq)
        attention_vq = attention_vq.masked_fill(softmax_mask_, 0.)

        # attention = self.dropout(attention)
        output_qv = torch.bmm(attention_qv, video)
        output_qv = self.linear_final_qv(torch.cat((output_qv, q), dim=-1))
        # output_q = self.layer_norm_qv(output_qv + q)
        output_q = self.layer_norm_qv(output_qv)

        output_vq = torch.bmm(attention_vq, question)
        output_vq = self.linear_final_vq(torch.cat((output_vq, v), dim=-1))
        # output_v = self.layer_norm_vq(output_vq + v)
        output_v = self.layer_norm_vq(output_vq)
        return output_q, output_v


class CoConcatAttention(nn.Module):

    def __init__(self, hidden_size, n_layers=1, dropout_p=0.0):
        super().__init__()

        self.encoder_layers = nn.ModuleList(
            [
                CoConcatAttentionLayer(hidden_size, dropout_p)
                for _ in range(n_layers)
            ])

    def forward(self, question, video):
        attn_mask = padding_mask_k(question, video)
        softmax_mask = padding_mask_q(question, video)
        attn_mask_ = padding_mask_k(video, question)
        softmax_mask_ = padding_mask_q(video, question)

        for encoder in self.encoder_layers:
            question, video = encoder(
                question,
                video,
                attn_mask=attn_mask,
                softmax_mask=softmax_mask,
                attn_mask_=attn_mask_,
                softmax_mask_=softmax_mask_)

        return question, video


class CoSiameseAttentionLayer(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)

        self.linear_question = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)
        self.linear_video = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)
        self.linear_v_question = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)
        self.linear_v_video = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)

        self.linear_final = nn.Sequential(
            nlpnn.WeightDropLinear(
                2 * hidden_size,
                hidden_size,
                weight_dropout=dropout_p,
                bias=False), nn.ReLU(),
            nlpnn.WeightDropLinear(
                hidden_size, hidden_size, weight_dropout=dropout_p, bias=False))

        self.layer_norm_qv = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.layer_norm_vq = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(
            self,
            question,
            video,
            scale=None,
            attn_mask=None,
            softmax_mask=None,
            attn_mask_=None,
            softmax_mask_=None):
        """
        Args:
            q: [B, L_q, D_q]
            k: [B, L_k, D_k]
            v: [B, L_v, D_v]
        """
        q = question
        v = video

        if attn_mask is None or softmax_mask is None:
            attn_mask = padding_mask_k(question, video)
            softmax_mask = padding_mask_q(question, video)
        if attn_mask_ is None or softmax_mask_ is None:
            attn_mask_ = padding_mask_k(video, question)
            softmax_mask_ = padding_mask_q(video, question)

        # linear projection
        question_q = self.linear_question(question)
        video_k = self.linear_video(video)
        question = self.linear_v_question(question)
        video = self.linear_v_video(video)

        scale = video.size(-1)**-0.5

        attention_qv = torch.bmm(question_q, video_k.transpose(1, 2))
        if scale is not None:
            attention_qv = attention_qv * scale
        if attn_mask is not None:
            attention_qv = attention_qv.masked_fill(attn_mask, -np.inf)
        attention_qv = self.softmax(attention_qv)
        attention_qv = attention_qv.masked_fill(softmax_mask, 0.)

        attention_vq = torch.bmm(video_k, question_q.transpose(1, 2))
        if scale is not None:
            attention_vq = attention_vq * scale
        if attn_mask_ is not None:
            attention_vq = attention_vq.masked_fill(attn_mask_, -np.inf)
        attention_vq = self.softmax(attention_vq)
        attention_vq = attention_vq.masked_fill(softmax_mask_, 0.)

        # attention = self.dropout(attention)
        output_qv = torch.bmm(attention_qv, video)
        output_qv = self.linear_final(torch.cat((output_qv, q), dim=-1))
        # output_q = self.layer_norm_qv(output_qv + q)
        output_q = self.layer_norm_qv(output_qv)

        output_vq = torch.bmm(attention_vq, question)
        output_vq = self.linear_final(torch.cat((output_vq, v), dim=-1))
        # output_v = self.layer_norm_vq(output_vq + v)
        output_v = self.layer_norm_vq(output_vq)
        return output_q, output_v


class CoSiameseAttention(nn.Module):

    def __init__(self, hidden_size, n_layers=1, dropout_p=0.0):
        super().__init__()

        self.encoder_layers = nn.ModuleList(
            [
                CoSiameseAttentionLayer(hidden_size, dropout_p)
                for _ in range(n_layers)
            ])

    def forward(self, question, video):
        attn_mask = padding_mask_k(question, video)
        softmax_mask = padding_mask_q(question, video)
        attn_mask_ = padding_mask_k(video, question)
        softmax_mask_ = padding_mask_q(video, question)

        for encoder in self.encoder_layers:
            question, video = encoder(
                question,
                video,
                attn_mask=attn_mask,
                softmax_mask=softmax_mask,
                attn_mask_=attn_mask_,
                softmax_mask_=softmax_mask_)

        return question, video


class SingleAttentionLayer(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)

        self.linear_q = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)
        self.linear_v = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)
        self.linear_k = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)

        self.linear_final = nlpnn.WeightDropLinear(
            hidden_size, hidden_size, weight_dropout=dropout_p, bias=False)

        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, q, k, v, scale=None, attn_mask=None, softmax_mask=None):
        """
        Args:
            q: [B, L_q, D_q]
            k: [B, L_k, D_k]
            v: [B, L_v, D_v]
        Return: Same shape to q, but in 'v' space, soft knn
        """

        if attn_mask is None or softmax_mask is None:
            attn_mask = padding_mask_k(q, k)
            softmax_mask = padding_mask_q(q, k)

        # linear projection
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        scale = v.size(-1)**-0.5

        attention = torch.bmm(q, k.transpose(-2, -1))
        if scale is not None:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = attention.masked_fill(softmax_mask, 0.)

        # attention = self.dropout(attention)
        output = torch.bmm(attention, v)
        output = self.linear_final(output)
        output = self.layer_norm(output + q)

        return output


class SingleAttention(nn.Module):

    def __init__(self, hidden_size, n_layers=1, dropout_p=0.0):
        super().__init__()

        self.encoder_layers = nn.ModuleList(
            [
                SingleAttentionLayer(hidden_size, dropout_p)
                for _ in range(n_layers)
            ])

    def forward(self, q, v):
        attn_mask = padding_mask_k(q, v)
        softmax_mask = padding_mask_q(q, v)

        for encoder in self.encoder_layers:
            q = encoder(q, v, v, attn_mask=attn_mask, softmax_mask=softmax_mask)

        return q


class SoftKNN(nn.Module):

    def __init__(self, model_dim=512, num_heads=1, dropout=0.0):
        super().__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(
            model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(
            model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(
            model_dim, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = ScaledDotProductAttention(dropout)

    def forward(self, query, key, value, attn_mask=None):

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads)**-0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        output = context.view(batch_size, -1, dim_per_head * num_heads)

        return output, attention


class CrossoverTransformerLayer(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super().__init__()

        self.v_transformer = MultiHeadAttention(model_dim, num_heads, dropout)
        self.q_transformer = MultiHeadAttention(model_dim, num_heads, dropout)
        self.v_feed_forward = PositionalWiseFeedForward(
            model_dim, ffn_dim, dropout)
        self.q_feed_forward = PositionalWiseFeedForward(
            model_dim, ffn_dim, dropout)

    def forward(self, question, video, q_mask=None, v_mask=None):
        # self attention, v_attention of shape (bs, v_len, q_len)
        video_, v_attention = self.v_transformer(
            video, question, question, v_mask)
        # feed forward network
        video_ = self.v_feed_forward(video_)

        # self attention, q_attention of shape (bs, q_len, v_len)
        question_, q_attention = self.q_transformer(
            question, video, video, q_mask)
        # feed forward network
        question_ = self.q_feed_forward(question_)

        return video_, question_, v_attention, q_attention


class CrossoverTransformer(nn.Module):

    def __init__(
            self,
            q_max_len=35,
            v_max_len=80,
            num_layers=2,
            model_dim=512,
            num_heads=8,
            ffn_dim=2048,
            dropout=0.0):
        super().__init__()

        self.encoder_layers = nn.ModuleList(
            [
                CrossoverTransformerLayer(
                    model_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ])

        # max_seq_len is 35 or 80
        self.q_pos_embedding = PositionalEncoding(model_dim, q_max_len)
        self.v_pos_embedding = PositionalEncoding(model_dim, v_max_len)

    def forward(self, question, video, q_length, v_length):
        # q_length of shape (batch, ), each item is the length of the seq
        question += self.q_pos_embedding(q_length)[:, :question.size()[1], :]
        video += self.v_pos_embedding(v_length)[:, :video.size()[1], :]

        # q_attention_mask of shape (bs, q_len, v_len)
        q_attention_mask = padding_mask_k(question, video)
        # v_attention_mask of shape (bs, v_len, q_len)
        v_attention_mask = padding_mask_k(video, question)

        q_attentions = []
        v_attentions = []
        for encoder in self.encoder_layers:
            video, question, v_attention, q_attention = encoder(
                question, video, q_attention_mask, v_attention_mask)
            q_attentions.append(q_attention)
            v_attentions.append(v_attention)

        return question, video, q_attentions, v_attentions


class MaskedCrossoverTransformerLayer(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super().__init__()

        self.v_transformer = MaskedMultiHeadAttention(
            model_dim, num_heads, dropout)
        self.q_transformer = MaskedMultiHeadAttention(
            model_dim, num_heads, dropout)
        self.v_feed_forward = MaskedPositionalWiseFeedForward(
            model_dim, ffn_dim, dropout)
        self.q_feed_forward = MaskedPositionalWiseFeedForward(
            model_dim, ffn_dim, dropout)

    def forward(
            self,
            question,
            video,
            q_mask=None,
            v_mask=None,
            q_sf_mask=None,
            v_sf_mask=None):
        # self attention, v_attention of shape (bs, v_len, q_len)
        video_, v_attention = self.v_transformer(
            video, question, question, v_mask, v_sf_mask)
        # feed forward network
        video_ = self.v_feed_forward(video_)

        # self attention, q_attention of shape (bs, q_len, v_len)
        question_, q_attention = self.q_transformer(
            question, video, video, q_mask, q_sf_mask)
        # feed forward network
        question_ = self.q_feed_forward(question_)

        return video_, question_, v_attention, q_attention


class MaskedCrossoverTransformer(nn.Module):

    def __init__(
            self,
            q_max_len=35,
            v_max_len=80,
            num_layers=2,
            model_dim=512,
            num_heads=8,
            ffn_dim=2048,
            dropout=0.0,
            position=False):
        super().__init__()

        self.position = position

        self.encoder_layers = nn.ModuleList(
            [
                MaskedCrossoverTransformerLayer(
                    model_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ])

        # max_seq_len is 35 or 80
        self.q_pos_embedding = PositionalEncoding(model_dim, q_max_len)
        self.v_pos_embedding = PositionalEncoding(model_dim, v_max_len)

    def forward(self, question, video, q_length, v_length):
        # q_length of shape (batch, ), each item is the length of the seq
        if self.position:
            question += self.q_pos_embedding(
                q_length)[:, :question.size()[1], :]
            video += self.v_pos_embedding(v_length)[:, :video.size()[1], :]

        q_attention_mask = padding_mask_k(question, video)
        q_softmax_mask = padding_mask_q(question, video)
        v_attention_mask = padding_mask_k(video, question)
        v_softmax_mask = padding_mask_q(video, question)

        q_attentions = []
        v_attentions = []
        for encoder in self.encoder_layers:
            video, question, v_attention, q_attention = encoder(
                question, video, q_attention_mask, v_attention_mask,
                q_softmax_mask, v_softmax_mask)
            q_attentions.append(q_attention)
            v_attentions.append(v_attention)

        return question, video, q_attentions, v_attentions


class SelfTransformerEncoder(nn.Module):

    def __init__(
            self,
            hidden_size,
            n_layers,
            dropout_p,
            vocab_size,
            q_max_len,
            v_max_len,
            embedding=None,
            update_embedding=True,
            position=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.ln_q = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_v = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.n_layers = n_layers
        self.position = position

        embedding_dim = embedding.shape[
            1] if embedding is not None else hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # ! no embedding init
        # if embedding is not None:
        #     # self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        #     self.embedding.weight = nn.Parameter(
        #         torch.from_numpy(embedding).float())
        self.upcompress_embedding = nlpnn.WeightDropLinear(
            embedding_dim, hidden_size, weight_dropout=dropout_p, bias=False)
        self.embedding.weight.requires_grad = update_embedding

        self.project_c3d = nlpnn.WeightDropLinear(4096, 2048, bias=False)

        self.project_resnet_and_c3d = nlpnn.WeightDropLinear(
            4096, hidden_size, weight_dropout=dropout_p, bias=False)

        # max_seq_len is 35 or 80
        self.q_pos_embedding = PositionalEncoding(hidden_size, q_max_len)
        self.v_pos_embedding = PositionalEncoding(hidden_size, v_max_len)

    def forward(self, question, resnet, c3d, q_length, v_length):
        ### question
        embedded = self.embedding(question)
        embedded = self.dropout(embedded)
        question = F.relu(self.upcompress_embedding(embedded))

        ### video
        # ! no relu
        c3d = self.project_c3d(c3d)
        video = F.relu(
            self.project_resnet_and_c3d(torch.cat((resnet, c3d), dim=2)))

        ### position encoding
        if self.position:
            question += self.q_pos_embedding(
                q_length)[:, :question.size()[1], :]
            video += self.v_pos_embedding(v_length)[:, :video.size()[1], :]

        # question = self.ln_q(question)
        # video = self.ln_v(video)
        return question, video
