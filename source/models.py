import torch
import torch.nn as nn

from layers import Pooling, Concat, GCN, EventLSTM, MLP, TimeAttention


###########################################
# TEMPORAL GCN ARCHITECTURES ##############
###########################################

class DyGED_CT(nn.Module):
    def __init__(self, adj_size, n_feature, n_hidden_gcn, n_hidden_mlp, n_output, k, dropout, pooling_key):
        super(DyGED_CT, self).__init__()

        # gcn layer
        self.gcn = GCN(adj_size, n_feature, n_hidden_gcn, dropout)

        # pooling layer
        self.pooling = Pooling(pooling_key, adj_size, n_hidden_gcn[-1])

        # concat layer
        self.concat = Concat(k + 1)

        # classification
        self.classification = MLP(self.pooling.get_dimension() * (k + 1), n_hidden_mlp, n_output, dropout=dropout)

    def forward(self, X, A):
        X = self.gcn(X, A)
        X, S_node = self.pooling(X)
        X = self.concat(X)
        outs = self.classification(X)
        return outs, X, S_node, None


class DyGED_NL(nn.Module):
    def __init__(self, adj_size, n_feature, n_hidden_gcn, attention_expert, n_hidden_mlp, n_output, k, dropout, pooling_key):
        super(DyGED_NL, self).__init__()

        # gcn layer
        self.gcn = GCN(adj_size, n_feature, n_hidden_gcn, dropout)

        # pooling layer
        self.pooling = Pooling(pooling_key, adj_size, n_hidden_gcn[-1])

        # attention layer
        self.attention = TimeAttention(self.pooling.get_dimension(), k + 1, attention_expert)

        # classification
        self.classification = MLP(self.attention.get_dimension() * attention_expert, n_hidden_mlp, n_output, dropout=dropout)

    def forward(self, X, A):
        X = self.gcn(X, A)
        X, S_node = self.pooling(X)
        X, S_time = self.attention(X)
        outs = self.classification(X)
        return outs, X, S_node, S_time


class DyGED_NA(nn.Module):
    def __init__(self, adj_size, n_feature, n_hidden_gcn, n_hidden_lstm, n_hidden_mlp, n_output, dropout, pooling_key):
        super(DyGED_NA, self).__init__()

        # gcn layer
        self.gcn = GCN(adj_size, n_feature, n_hidden_gcn, dropout)

        # pooling layer
        self.pooling = Pooling(pooling_key, adj_size, n_hidden_gcn[-1])

        # lstm layer
        self.lstm = EventLSTM(input_size=self.pooling.get_dimension(), hidden_size=n_hidden_lstm, dropout=dropout)

        # classification
        self.classification = MLP(self.lstm.get_dimension(), n_hidden_mlp, n_output, dropout=dropout)

    def forward(self, X, A):
        X = self.gcn(X, A)
        X, S_node = self.pooling(X)
        X = self.lstm(X)
        outs = self.classification(X)
        return outs, X, S_node, None


class DyGED(nn.Module):
    def __init__(self, adj_size, n_feature, n_hidden_gcn, n_hidden_lstm, attention_expert, n_hidden_mlp, n_output, k, dropout, pooling_key):
        super(DyGED, self).__init__()

        # gcn layer
        self.gcn = GCN(adj_size, n_feature, n_hidden_gcn, dropout)

        # pooling layer
        self.pooling = Pooling(pooling_key, adj_size, n_hidden_gcn[-1])

        # lstm layer
        self.lstm = EventLSTM(input_size=self.pooling.get_dimension(), hidden_size=n_hidden_lstm, dropout=dropout)

        # attention layer
        self.attention = TimeAttention(self.lstm.get_dimension(), k + 1, attention_expert)

        # classification
        self.classification = MLP(self.attention.get_dimension(), n_hidden_mlp, n_output, dropout=dropout)

    def forward(self, X, A):
        X = self.gcn(X, A)
        X, S_node = self.pooling(X)
        X = self.lstm(X)
        X, S_time = self.attention(X)
        outs = self.classification(X)
        return outs, X, S_node, S_time


model = DyGED(adj_size=50, n_feature=32,
              n_hidden_gcn=(32,), n_hidden_mlp=(32,), n_hidden_lstm=(32,),
              attention_expert=1, n_output=2, k=3, dropout=0.1, pooling_key='expert')

# # debug data
# G = torch.randn((100, 50, 50))
# G[G < 0.5] = 0
# G[G >= 0.5] = 1
# X = torch.randn((100, 50, 32))
#
# outs, X, S_node, S_time = model(X, G)
#
# print(outs)
