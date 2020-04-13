import os
import random
import spacy
from argparse import Namespace

import torch
from torchtext import data
from torchtext import datasets
import torch.nn as nn
import torch.optim as optim

from models import BiLSTM
from training import run
from data_utils import (handle_dirs, pickle_dump, pickle_load)


SEED = 42
torch.manual_seed(SEED)
MAX_VOCAB_SIZE = 25_000
EMBEDDINGS_FILE = 'glove.6B.100d'

args = Namespace(
    # Data and Path hyper parameters
    data_file='data.pkl',
    model_state_file='imdb_model.torch',
    log_file='imdb.log',
    train_state_file='train_state.json',
    save_dir='.save/imdb/',
    # Model hyper parameters
    embedding_dim=100,
    hidden_dim=256,
    output_dim = 1,
    num_layers=2,
    # Training hyper parameter
    seed=42,
    learning_rate=0.001,
    dropout_p=0.5,
    batch_size=64,
    num_epochs=5,
    early_stopping_criteria=5,
    # Runtime option
    reload_from_files=False,
    expand_filepaths_to_save_dir=True,
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

if args.expand_filepaths_to_save_dir:
        args.data_file = os.path.join(args.save_dir,
                                      args.data_file)
        args.model_state_file = os.path.join(args.save_dir,
                                             args.model_state_file)
        args.log_file = os.path.join(args.save_dir,
                                     args.log_file)
        args.train_state_file = os.path.join(args.save_dir,
                                             args.train_state_file)
        print('Expanded filepaths: ')
        print(f'\t{args.data_file}')
        print(f'\t{args.model_state_file}')
        print(f'\t{args.log_file}')
        print(f'\t{args.train_state_file}')

handle_dirs(args.save_dir)

if args.reload_from_files:
    text_field = data.Field(tokenize='spacy', include_lengths=True)
    label_field = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(text_field, label_field)
    train_data, valid_data = train_data.split(random_state = random.seed(SEED))

    text_field, vocab_field = pickle_load(args.data_file)
    PAD_IDX = text_field.vocab.stoi[text_field.pad_token]
    print('Vocabularies and embeddings reloaded...')
else:
    text_field = data.Field(tokenize='spacy', include_lengths=True)
    label_field = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(text_field, label_field)
    train_data, valid_data = train_data.split(random_state = random.seed(SEED))

    text_field.build_vocab(train_data, 
                        max_size = MAX_VOCAB_SIZE, 
                        vectors = EMBEDDINGS_FILE, 
                        unk_init = torch.Tensor.normal_)

    label_field.build_vocab(train_data)

    PAD_IDX = text_field.vocab.stoi[text_field.pad_token]
    UNK_IDX = text_field.vocab.stoi[text_field.unk_token]
    pretrained_embeddings = text_field.vocab.vectors
    pretrained_embeddings[UNK_IDX] = torch.zeros(args.embedding_dim)
    pretrained_embeddings[PAD_IDX] = torch.zeros(args.embedding_dim)

    obj = (text_field, label_field)
    pickle_dump(obj, args.data_file)



train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=args.batch_size,
    sort_within_batch=True,
    device=args.device)

model = BiLSTM(
    args.embedding_dim, 
    args.hidden_dim, 
    args.output_dim, 
    args.num_layers,
    pretrained_embeddings,
    args.dropout_p, 
    PAD_IDX
)

loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 mode='min', factor=0.5,
                                                 patience=1)

iterator = dict(train=train_iterator, test=test_iterator, valid=valid_iterator)
run(args, model, loss_func, optimizer, scheduler, iterator)