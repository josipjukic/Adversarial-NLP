import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN(nn.Module):

    def __init__(self, embedding_dim, num_embeddings,
                 pretrained_embeddings=None, padding_idx=0,
                 hidden_dim=100, output_dim=1, num_layers=1,
                 bidirectional=False, nonlinearity='tanh', dropout=0.):
        super(RNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

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

        self.rnn = nn.RNN(self.embedding_dim, self.hidden_dim, self.num_layers,
                          bidirectional=bidirectional, dropout=dropout,
                          nonlinearity=nonlinearity)

        hidden2out_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.linear = nn.Linear(hidden2out_dim, self.output_dim)

    def forward(self, x_in, apply_softmax=False):
        x_embedded = self.emb(x_in).permute(1, 0, 2)
        out, self.hidden = self.rnn(x_embedded)
        y_pred = self.linear(out[-1, :, :])

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        
        return y_pred


class LSTM(nn.Module):

    def __init__(self, embedding_dim, num_embeddings,
                 pretrained_embeddings=None, padding_idx=0,
                 hidden_dim=100, output_dim=1, num_layers=1,
                 bidirectional=False, dropout=0.):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

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

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers,
                            bidirectional=bidirectional, dropout=dropout)

        hidden2out_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.linear = nn.Linear(hidden2out_dim, self.output_dim)

    def forward(self, x_in, apply_softmax=False):
        x_embedded = self.emb(x_in).permute(1, 0, 2)
        out, self.hidden = self.lstm(x_embedded)
        y_pred = self.linear(out[-1, :, :])

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        return y_pred
