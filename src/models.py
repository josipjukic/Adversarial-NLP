import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_layers, pretrained_embeddings, bidirectional,
                 dropout_p=0., padding_idx=0, nonlinearity='tanh'):
        
        super().__init__()
        self.output_dim = output_dim
        self.bidirectional = bidirectional

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

    def predict_proba(self, x_in):
        self.eval()
        y_pred = self.forward(x_in)
        if self.output_dim == 1:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = F.softmax
        return y_pred

    def predict(self, x_in):
        self.eval()
        y_pred = self.forward(x_in)
        if self.output_dim == 1:
            y_pred = torch.sigmoid(y_pred)
            out = torch.as_tensor((y_pred - 0.5) > 0, dtype=torch.long)  
        else:
            y_pred = F.softmax(y_pred, dim=1)
            out = torch.argmax(y_pred, dim=1)
        return out
        

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_layers, pretrained_embeddings, bidirectional,
                 dropout_p=0., padding_idx=0):
        
        super().__init__()
        self.output_dim = output_dim
        self.bidirectional = bidirectional

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

    def predict_proba(self, x_in):
        self.eval()
        y_pred = self.forward(x_in)
        if self.output_dim == 1:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = F.softmax
        return y_pred

    def predict(self, x_in):
        self.eval()
        y_pred = self.forward(x_in)
        if self.output_dim == 1:
            y_pred = torch.sigmoid(y_pred)
            out = torch.as_tensor((y_pred - 0.5) > 0, dtype=torch.long)  
        else:
            y_pred = F.softmax(y_pred, dim=1)
            out = torch.argmax(y_pred, dim=1)
        return out


class PackedRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_layers, pretrained_embeddings, bidirectional,
                 dropout_p=0., padding_idx=0, nonlinearity='tanh'):
        
        super().__init__()
        self.output_dim = output_dim
        self.bidirectional = bidirectional

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

    def predict_proba(self, batch):
        x_in, lengths = batch
        self.eval()
        y_pred = self.forward(x_in, lengths)
        if self.output_dim == 1:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = F.softmax
        return y_pred

    def predict(self, batch):
        x_in, lengths = batch
        self.eval()
        y_pred = self.forward(x_in, lengths)
        if self.output_dim == 1:
            y_pred = torch.sigmoid(y_pred)
            out = torch.as_tensor((y_pred - 0.5) > 0, dtype=torch.long)  
        else:
            y_pred = F.softmax(y_pred, dim=1)
            out = torch.argmax(y_pred, dim=1)
        return out


class PackedLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_layers, pretrained_embeddings, bidirectional,
                 dropout_p=0., padding_idx=0):
        
        super().__init__()
        self.output_dim = output_dim
        self.bidirectional = bidirectional

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

    def predict_proba(self, batch):
        x_in, lengths = batch
        self.eval()
        y_pred = self.forward(x_in, lengths)
        if self.output_dim == 1:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = F.softmax
        return y_pred

    def predict(self, batch):
        x_in, lengths = batch
        self.eval()
        y_pred = self.forward(x_in, lengths)
        if self.output_dim == 1:
            y_pred = torch.sigmoid(y_pred)
            out = torch.as_tensor((y_pred - 0.5) > 0, dtype=torch.long)  
        else:
            y_pred = F.softmax(y_pred, dim=1)
            out = torch.argmax(y_pred, dim=1)
        return out
