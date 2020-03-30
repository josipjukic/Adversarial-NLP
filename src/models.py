import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2,
                 bidirectional=False, nonlinearity='tanh', dropout=0.):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

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

    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, bidirectional=False, dropout=0.):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,
                            bidirectional=bidirectional, dropout=dropout)

        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x_in, apply_softmax=False):
        lstm_out, self.hidden = self.lstm(x_in)
        y_pred = self.linear(lstm_out)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        return y_pred