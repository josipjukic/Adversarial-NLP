from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

RNN_TYPES = ['RNN', 'LSTM', 'GRU']


class AbstractModel(ABC):
    def predict_proba(self, batch):
        self.eval()
        y_pred = self.forward(batch)
        if self.output_dim == 1:
            y_pred = torch.sigmoid(y_pred)
            y_pred = torch.cat([1.-y_pred, y_pred], dim=1)
        else:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred

    def predict(self, batch):
        self.eval()
        y_pred = self.forward(batch)
        if self.output_dim == 1:
            y_pred = torch.sigmoid(y_pred)
            out = torch.as_tensor((y_pred - 0.5) > 0,
                                  dtype=torch.long, device=self.device)
        else:
            y_pred = F.softmax(y_pred, dim=1)
            out = torch.argmax(y_pred, dim=1)
        return out


class RNN(nn.Module, AbstractModel):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_layers, pretrained_embeddings, bidirectional,
                 dropout_p=0., padding_idx=0, rnn_type='RNN',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        super().__init__()
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.device = device

        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings,
                                                      padding_idx=padding_idx)

        drop_prob = 0. if num_layers > 1 else dropout_p
        assert rnn_type in RNN_TYPES, f'Use one of the following: {str(RNN_TYPES)}'
        RnnCell = getattr(nn, rnn_type)
        self.rnn = RnnCell(embedding_dim,
                           hidden_dim,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=drop_prob)

        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, batch):
        # x_in: S x B
        x_in = batch.text

        # embedded: S x B x E
        embedded = self.dropout(self.embedding(x_in))

        # out: S x B x (H*num_directions)
        # hidden: (L*num_directions) x B x H
        out, hidden = self.rnn(embedded)
        if type(hidden) == tuple: hidden = hidden[0]

        # if bidirectional concat the final forward (hidden[-2,:,:]) and
        # backward (hidden[-1,:,:]) hidden layers, otherwise extract the
        # final hidden state and apply dropout
        # hidden = B x (H*num_directions)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        hidden = self.dropout(hidden)
        return self.fc(hidden)


class PackedRNN(RNN):
    def forward(self, batch):
        # x_in: S x B
        x_in, lengths = batch.text

        # embedded: S x B x E
        embedded = self.dropout(self.embedding(x_in))

        # pack sequence
        # output over padding tokens are zero tensors
        # hidden: (L*num_directions) x B x H
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        packed_out, hidden = self.rnn(packed_embedded)
        if type(hidden) == tuple: hidden = hidden[0]

        # unpack sequence
        # out: S x B x (H*num_directions)
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_out)

        # if bidirectional concat the final forward (hidden[-2,:,:]) and
        # backward (hidden[-1,:,:]) hidden layers, otherwise extract the
        # final hidden state and apply dropout
        # hidden = B x (H*num_directions)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        hidden = self.dropout(hidden)
        return self.fc(hidden)


class BertClassifier(nn.Module, AbstractModel):
    def __init__(self, output_dim, bert_model_name='bert-base-uncased'):
        super().__init__()
        self.output_dim = output_dim
        self.bert = BertModel.from_pretrained(bert_model_name)
        config = self.bert.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, output_dim)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, batch):
        x_in = batch.text
        _, pooled_output = self.bert(x_in.permute(1, 0))
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits


class RNN_NLI(nn.Module, AbstractModel):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_layers, pretrained_embeddings, bidirectional,
                 dropout_p=0., padding_idx=0, rnn_type='LSTM',
                 num_out_layers=4,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        super().__init__()
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.device = device

        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings,
                                                      padding_idx=padding_idx)

        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)

        drop_prob = 0. if num_layers > 1 else dropout_p
        assert rnn_type in RNN_TYPES, f'Use one of the following: {str(RNN_TYPES)}'
        RnnCell = getattr(nn, rnn_type)
        self.rnn = RnnCell(embedding_dim,
                           hidden_dim,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=drop_prob)

        self.relu = nn.ReLU()

        out_layers = []
        lin_dim = hidden_dim*2
        if bidirectional: lin_dim *= 2
        for _ in range(num_out_layers-1):
            out_layers.append(nn.Linear(lin_dim, lin_dim))
            out_layers.append(self.relu)
            out_layers.append(self.dropout)
        out_layers.append(nn.Linear(lin_dim, output_dim))
        self.out = nn.Sequential(*out_layers)

    def forward(self, batch):
        premise_embed = self.embedding(batch.premise)
        hypothesis_embed = self.embedding(batch.hypothesis)
        premise_proj = self.relu(self.projection(premise_embed))
        hypothesis_proj = self.relu(self.projection(hypothesis_embed))
        encoded_premise, _ = self.rnn(premise_proj)
        encoded_hypothesis, _ = self.rnn(hypothesis_proj)
        premise = encoded_premise.sum(dim=0)
        hypothesis = encoded_hypothesis.sum(dim=0)
        combined = torch.cat((premise, hypothesis), 1)
        return self.out(combined)
