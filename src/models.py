import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC


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
            out = torch.as_tensor((y_pred - 0.5) > 0, dtype=torch.long, device=self.device)  
        else:
            y_pred = F.softmax(y_pred, dim=1)
            out = torch.argmax(y_pred, dim=1)
        return out


class RNN(nn.Module, AbstractModel):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_layers, pretrained_embeddings, bidirectional,
                 dropout_p=0., padding_idx=0, nonlinearity='tanh',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        super().__init__()
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.device = device

        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings,
                                                      padding_idx=padding_idx)

        drop_prob = 0. if num_layers > 1 else dropout_p
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers,
                          bidirectional=bidirectional, nonlinearity=nonlinearity,
                          dropout=drop_prob)

        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x_in):
        # x_in: S x B
        # embedded: S x B x E
        embedded = self.dropout(self.embedding(x_in))

        # out: S x B x (H*num_directions)
        # hidden: (L*num_directions) x B x H
        # cell: (L*num_directions) x B x H
        out, (hidden, cell) = self.rnn(embedded)

        # if bidirectional concat the final forward (hidden[-2,:,:]) and
        # backward (hidden[-1,:,:]) hidden layers, otherwise extract the
        # final hidden state and apply dropout
        # hidden = B x (H*num_directions)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        hidden = self.dropout(hidden)
        return self.fc(hidden)


class LSTM(nn.Module, AbstractModel):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_layers, pretrained_embeddings, bidirectional,
                 dropout_p=0., padding_idx=0,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        super().__init__()
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.device = device

        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings,
                                                      padding_idx=padding_idx)
        
        drop_prob = 0. if num_layers > 1 else dropout_p
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional, 
                            dropout=drop_prob)

        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x_in):
        # x_in: S x B
        # embedded: S x B x E
        embedded = self.dropout(self.embedding(x_in))
        
        # out: S x B x (H*num_directions)
        # hidden: (L*num_directions) x B x H
        # cell: (L*num_directions) x B x H
        out_out, (hidden, cell) = self.lstm(embedded)
        

        # if bidirectional concat the final forward (hidden[-2,:,:]) and
        # backward (hidden[-1,:,:]) hidden layers, otherwise extract the
        # final hidden state and apply dropout
        # hidden = B x (H*num_directions)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        hidden = self.dropout(hidden)
        return self.fc(hidden)


class PackedRNN(nn.Module, AbstractModel):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_layers, pretrained_embeddings, bidirectional,
                 dropout_p=0., padding_idx=0, nonlinearity='tanh',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        super().__init__()
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.device = device

        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings,
                                                      padding_idx=padding_idx)

        drop_prob = 0. if num_layers > 1 else dropout_p
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers,
                          bidirectional=bidirectional, nonlinearity=nonlinearity,
                          dropout=drop_prob)

        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, batch):
        x_in, lengths = batch
        # x_in: S x B
        # embedded: S x B x E
        embedded = self.dropout(self.embedding(x_in))
       
        # pack sequence
        # output over padding tokens are zero tensors
        # hidden: (L*num_directions) x B x H
        # cell: (L*num_directions) x B x H
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        packed_out, (hidden, cell) = self.rnn(packed_embedded)
        
        # unpack sequence
        # out: S x B x (H*num_directions)
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_out)

        # if bidirectional concat the final forward (hidden[-2,:,:]) and
        # backward (hidden[-1,:,:]) hidden layers, otherwise extract the
        # final hidden state and apply dropout
        # hidden = B x (H*num_directions)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        hidden = self.dropout(hidden)
        return self.fc(hidden)


class PackedLSTM(nn.Module, AbstractModel):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_layers, pretrained_embeddings, bidirectional,
                 dropout_p=0., padding_idx=0,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        super().__init__()
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.device = device

        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings,
                                                      padding_idx=padding_idx)
        
        drop_prob = 0. if num_layers > 1 else dropout_p
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional, 
                            dropout=drop_prob)

        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, batch):
        x_in, lengths = batch
        # x_in: S x B
        # embedded: S x B x E
        embedded = self.dropout(self.embedding(x_in))
        
        # pack sequence
        # output over padding tokens are zero tensors
        # hidden: (L*num_directions) x B x H
        # cell: (L*num_directions) x B x H
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        packed_out, (hidden, cell) = self.lstm(packed_embedded)
        
        # unpack sequence
        # out: S x B x (H*num_directions)
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_out)

        # if bidirectional concat the final forward (hidden[-2,:,:]) and
        # backward (hidden[-1,:,:]) hidden layers, otherwise extract the
        # final hidden state and apply dropout
        # hidden = B x (H*num_directions)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
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

    def forward(self, x_in):
        _, pooled_output = self.bert(x_in)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits