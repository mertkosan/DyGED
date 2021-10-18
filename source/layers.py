import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F


###########################################
# FORWARD PROPAGATION NEURAL NET LAYERS ###
###########################################

class MLP(nn.Module):
    def __init__(self, n_feature, n_hidden: tuple, n_output, dropout: float):
        super(MLP, self).__init__()

        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.layer_sizes = list(n_hidden) + [n_output]

        current_feature = n_feature
        for i in range(len(self.layer_sizes) - 1):
            layer = nn.Linear(current_feature, self.layer_sizes[i])
            self.layers.append(layer)
            current_feature = self.layer_sizes[i]
            self.batch_norms.append(nn.BatchNorm1d(current_feature))
        self.out_layer = nn.Linear(current_feature, n_output)

        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.Sigmoid()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.batch_norms[i](x)
            x = self.act(x)
            x = self.dropout(x)
        x = self.out_layer(x)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


################################
# GRAPH CONVOLUTIONAL LAYERS ###
################################

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X, A):
        aggregation = torch.matmul(A, X)
        output = torch.matmul(aggregation, self.weight)
        return output + self.bias if self.bias is not None else output


class GCN(nn.Module):
    def __init__(self, adj_size, in_features, n_hidden, dropout, bias=True):
        super(GCN, self).__init__()
        self.gcn_layers = nn.ModuleList()
        current_embedding_size = in_features
        for i in range(len(n_hidden)):
            layer = GraphConvolution(current_embedding_size, n_hidden[i], bias=bias)
            self.gcn_layers.append(layer)
            current_embedding_size = n_hidden[i]

        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(adj_size)

    def forward(self, X, A):
        embeddings = X
        for gcn_layer in self.gcn_layers:
            embeddings = gcn_layer(embeddings, A)
            embeddings = self.batch_norm(embeddings)
            embeddings = self.act(embeddings)
            embeddings = self.dropout(embeddings)
        return embeddings


################################
# POOLING LAYERS ###############
################################

class Pooling(nn.Module):
    def __init__(self, pooling_key, adj_size, embedding_size):
        super(Pooling, self).__init__()
        self.pooling_key = pooling_key

        self.dimension = None
        if self.pooling_key == 'mean':
            self.dimension = embedding_size
        elif self.pooling_key == 'max':
            self.dimension = embedding_size
        elif self.pooling_key == 'expert':
            number_of_expert = 1
            self.dimension = embedding_size * number_of_expert
            self.w1 = nn.Linear(embedding_size, embedding_size)
            self.w2 = nn.Linear(embedding_size, number_of_expert)
        else:
            raise NotImplementedError(f'Pooling operator {self.pooling_key} is not implemented yet!')

    def get_dimension(self):
        return self.dimension

    def forward(self, X):
        if self.pooling_key == 'mean':
            return X.mean(dim=-2), None
        elif self.pooling_key == 'max':
            return X.max(dim=-2)[0], None
        elif self.pooling_key == 'expert':
            S = torch.softmax(self.w2(torch.tanh(self.w1(X))), dim=1)
            return torch.matmul(torch.transpose(S, 1, 2), X).flatten(start_dim=1), S


################################
# CONCAT LAYERS ################
################################

class Concat(nn.Module):
    def __init__(self, window):
        super(Concat, self).__init__()
        self.window = window

    def forward(self, embeddings):
        return embeddings.unfold(0, self.window, 1).transpose(1, 2).flatten(start_dim=1)

        # slow
        # out = []
        # for i in range(self.window, len(embeddings) + 1):
        #     start = i - self.window
        #     end = i
        #     embedding = embeddings[start:end].flatten()
        #     out.append(embedding)
        # return torch.stack(out)


###########################################
# RECURRENT NEURAL NET LAYERS #############
###########################################

class EventLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(EventLSTM, self).__init__()

        self.num_layers = len(hidden_size)
        self.embeddings_size = hidden_size[0]
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.embeddings_size, num_layers=self.num_layers, dropout=dropout)

    def get_dimension(self):
        return self.embeddings_size

    def forward(self, embeddings):
        h_0 = torch.zeros((self.num_layers, embeddings.size(0), self.embeddings_size), device=embeddings.device)  # hidden state
        c_0 = torch.zeros((self.num_layers, embeddings.size(0), self.embeddings_size), device=embeddings.device)  # cell state

        output, (h_n, c_n) = self.lstm(embeddings.unsqueeze(0), (h_0, c_0))

        return output.squeeze(0)


class TimeAttention(nn.Module):
    def __init__(self, embeddings_size, window, expert_number):
        super(TimeAttention, self).__init__()

        self.window = window
        self.expert_number = expert_number
        self.embeddings_size = embeddings_size

        self.w1 = nn.Linear(embeddings_size, embeddings_size)
        self.w2 = nn.Linear(embeddings_size, expert_number)

    def get_dimension(self):
        return self.embeddings_size

    def forward(self, embeddings):
        embeddings = embeddings.unfold(0, self.window, 1).transpose(1, 2)
        S = torch.softmax(self.w2(torch.tanh(self.w1(embeddings))), dim=1)
        return torch.matmul(torch.transpose(S, 1, 2), embeddings).flatten(start_dim=1), S
