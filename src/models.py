import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN(nn.Module):

    def __init__(self, embedding_size, num_embeddings,
                 input_dim, hidden_dim, output_dim=1, num_layers=2,
                 pretrained_embeddings=None, padding_idx=0,
                 bidirectional=False, nonlinearity='tanh', dropout=0.):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        if pretrained_embeddings is None:
            self.emb = nn.Embedding(embedding_dim=embedding_dim,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx)
        else:
            emb = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(embedding_dim=embedding_dim,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx,
                                    _weight=emb)

        self.lstm = nn.RNN(self.input_dim, self.hidden_dim, self.num_layers,
                           bidirectional=bidirectional, dropout=dropout,
                           nonlinearity=nonlinearity)

        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x_in, apply_softmax=False):
        rnn_out, self.hidden = self.rnn(x_in)
        y_pred = self.linear(rnn_out)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        return y_pred


class LSTM(nn.Module):

    def __init__(self, embedding_size, num_embeddings,
                 pretrained_embeddings=None, padding_idx=0,
                 input_dim, hidden_dim, output_dim=1, num_layers=2,
                 bidirectional=False, dropout=0.):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        if pretrained_embeddings is None:
            self.emb = nn.Embedding(embedding_dim=embedding_dim,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx)
        else:
            emb = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(embedding_dim=embedding_dim,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx,
                                    _weight=emb)

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,
                            bidirectional=bidirectional, dropout=dropout)

        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x_in, apply_softmax=False):
        lstm_out, self.hidden = self.lstm(x_in)
        y_pred = self.linear(lstm_out)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        return y_pred
