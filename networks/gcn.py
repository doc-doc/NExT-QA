import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


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


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.layer_norm = nn.LayerNorm(out_features, elementwise_affine=False)

    def forward(self, input, adj):
        # self.weight of shape (hidden_size, hidden_size)
        support = self.weight(input)
        output = torch.bmm(adj, support)
        output = self.layer_norm(output)
        return output


class GraphAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(
            nn.init.xavier_normal_(
                torch.Tensor(in_features, out_features).type(
                    torch.cuda.FloatTensor if torch.cuda.is_available(
                    ) else torch.FloatTensor),
                gain=np.sqrt(2.0)),
            requires_grad=True)
        self.a1 = nn.Parameter(
            nn.init.xavier_normal_(
                torch.Tensor(out_features, 1).type(
                    torch.cuda.FloatTensor if torch.cuda.is_available(
                    ) else torch.FloatTensor),
                gain=np.sqrt(2.0)),
            requires_grad=True)
        self.a2 = nn.Parameter(
            nn.init.xavier_normal_(
                torch.Tensor(out_features, 1).type(
                    torch.cuda.FloatTensor if torch.cuda.is_available(
                    ) else torch.FloatTensor),
                gain=np.sqrt(2.0)),
            requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        f_1 = torch.matmul(h, self.a1)
        f_2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(f_1 + f_2.transpose(0, 1))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GCN(nn.Module):

    def __init__(
            self, input_size, hidden_size, num_classes, num_layers=1,
            dropout=0.1):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(input_size, hidden_size))
        for i in range(num_layers - 1):
            self.layers.append(GraphConvolution(hidden_size, hidden_size))
        self.layers.append(GraphConvolution(hidden_size, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = self.dropout(F.relu(layer(x, adj)))

        # x of shape (bs, q_v_len, num_classes)
        return x


class AdjLearner(Module):

    def __init__(self, in_feature_dim, hidden_size, dropout=0.1):
        super().__init__()
        '''
        ## Variables:
        - in_feature_dim: dimensionality of input features
        - hidden_size: dimensionality of the joint hidden embedding
        - K: number of graph nodes/objects on the image
        '''

        # Embedding layers. Padded 0 => 0
        self.edge_layer_1 = nn.Linear(in_feature_dim, hidden_size, bias=False)
        self.edge_layer_2 = nn.Linear(hidden_size, hidden_size, bias=False)

        # Regularisation
        self.dropout = nn.Dropout(p=dropout)
        self.edge_layer_1 = nn.utils.weight_norm(self.edge_layer_1)
        self.edge_layer_2 = nn.utils.weight_norm(self.edge_layer_2)

    def forward(self, questions, videos):
        '''
        ## Inputs:
        ## Returns:
        - adjacency matrix (batch_size, q_v_len, q_v_len)
        '''
        # graph_nodes (batch_size, q_v_len, in_feat_dim): input features
        graph_nodes = torch.cat((questions, videos), dim=1)

        # layer 1
        h = self.edge_layer_1(graph_nodes)
        h = F.relu(h)

        # layer 2
        h = self.edge_layer_2(h)
        h = F.relu(h)
        # h * sigmoid(Wh)
        # h = F.tanh(h)

        # outer product
        adjacency_matrix = torch.bmm(h, h.transpose(1, 2))

        return adjacency_matrix


class EvoAdjLearner(Module):

    def __init__(self, in_feature_dim, hidden_size, dropout=0.1):
        super().__init__()
        '''
        ## Variables:
        - in_feature_dim: dimensionality of input features
        - hidden_size: dimensionality of the joint hidden embedding
        - K: number of graph nodes/objects on the image
        '''

        # Embedding layers. Padded 0 => 0
        self.edge_layer_1 = nn.Linear(in_feature_dim, hidden_size, bias=False)
        self.edge_layer_2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.edge_layer_3 = nn.Linear(in_feature_dim, hidden_size, bias=False)
        self.edge_layer_4 = nn.Linear(hidden_size, hidden_size, bias=False)

        # Regularisation
        self.dropout = nn.Dropout(p=dropout)
        self.edge_layer_1 = nn.utils.weight_norm(self.edge_layer_1)
        self.edge_layer_2 = nn.utils.weight_norm(self.edge_layer_2)

    def forward(self, questions, videos):
        '''
        ## Inputs:
        ## Returns:
        - adjacency matrix (batch_size, q_v_len, q_v_len)
        '''
        # graph_nodes (batch_size, q_v_len, in_feat_dim): input features
        graph_nodes = torch.cat((questions, videos), dim=1)

        attn_mask = padding_mask_k(graph_nodes, graph_nodes)
        sf_mask = padding_mask_q(graph_nodes, graph_nodes)

        # layer 1
        h = self.edge_layer_1(graph_nodes)
        h = F.relu(h)
        # layer 2
        h = self.edge_layer_2(h)
        # h = F.relu(h)

        # layer 1
        h_ = self.edge_layer_3(graph_nodes)
        h_ = F.relu(h_)
        # layer 2
        h_ = self.edge_layer_4(h_)
        # h_ = F.relu(h_)

        # outer product
        adjacency_matrix = torch.bmm(h, h_.transpose(1, 2))
        # adjacency_matrix = adjacency_matrix.masked_fill(attn_mask, -np.inf)

        # softmaxed_adj = F.softmax(adjacency_matrix, dim=-1)

        # softmaxed_adj = softmaxed_adj.masked_fill(sf_mask, 0.)

        return adjacency_matrix