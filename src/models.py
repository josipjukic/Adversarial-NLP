import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_layers, pretrained_embeddings, bidirectional,
                 dropout_p=0., padding_idx=0):
        
        super().__init__()
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings,
                                                      padding_idx=padding_idx)
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           dropout=(0. if num_layers > 1 else dropout_p))

        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x_in, lengths):
        # text: S x B
        # embedded: S x B x E
        embedded = self.dropout(self.embedding(x_in))
       
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        packed_out, (hidden, cell) = self.rnn(packed_embedded)
        
        #unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_out)

        # out: S x B x (H*num_directions)
        # output over padding tokens are zero tensors
        
        # hidden: (L*num_directions) x B x H
        # cell: (L*num_directions) x B x H
        
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        
        # hidden = B x (H*num_directions)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        hidden = self.dropout(hidden)
        return self.fc(hidden)


class PlainRNN(nn.Module):

    def __init__(self, embedding_dim, num_embeddings,
                 pretrained_embeddings=None, padding_idx=0,
                 hidden_dim=100, output_dim=1, num_layers=1,
                 bidirectional=False, nonlinearity='tanh', dropout=0.):
        super().__init__()
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


class PlainLSTM(nn.Module):

    def __init__(self, embedding_dim, num_embeddings,
                 pretrained_embeddings=None, padding_idx=0,
                 hidden_dim=100, output_dim=1, num_layers=1,
                 bidirectional=False, dropout=0.):
        super().__init__()
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
