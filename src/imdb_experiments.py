import os
from argparse import Namespace
from functools import partial
import logging
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from vectorization import EmbeddingVectorizer
from vectorization import vectorizer_from_dataframe
from embeddings import GLOVE_6B_100D_PATH
from embeddings import make_embedding_matrix
from data_utils import (set_seed_everywhere, handle_dirs,
                        pickle_dump, pickle_load)
from data_utils import ImdbDataset
from models import LSTM
from training import run_experiment


if __name__ == '__main__':

    cwd = os.getcwd()
    root = os.path.join(cwd, '..')
    imdb_file_path = os.path.join(root, 'datasets/imdb/imdb_clean_split.csv')
    emb_file_path = os.path.join(root, GLOVE_6B_100D_PATH)
    save_path = os.path.join(root, 'save/imdb')

    args = Namespace(
        # Data and Path hyper parameters
        data_csv=imdb_file_path,
        vectorizer_file='imdb_vectorizer.pkl',
        embeddings_file='imdb_embeddings.pkl',
        model_state_file='imdb_model.torch',
        log_file='imdb.log',
        train_state_file='train_state.json',
        save_dir=save_path,
        # Model hyper parameters
        glove_filepath=emb_file_path,
        use_glove=True,
        embedding_dim=100,
        hidden_dim=256,
        num_layers=2,
        # Training hyper parameter
        seed=42,
        learning_rate=0.001,
        dropout=0.,
        batch_size=128,
        num_epochs=2,
        early_stopping_criteria=5,
        # Runtime option
        cuda=True,
        reload_from_files=True,
        expand_filepaths_to_save_dir=True)

    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)
        args.embeddings_file = os.path.join(args.save_dir,
                                            args.embeddings_file)
        args.model_state_file = os.path.join(args.save_dir,
                                             args.model_state_file)
        args.log_file = os.path.join(args.save_dir,
                                     args.log_file)
        args.train_state_file = os.path.join(args.save_dir,
                                             args.train_state_file)
        print("Expanded filepaths: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.embeddings_file))
        print("\t{}".format(args.model_state_file))
        print("\t{}".format(args.log_file))
        print("\t{}".format(args.train_state_file))

    # check CUDA
    if not torch.cuda.is_available():
        args.cuda = False

    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("Using CUDA: {}".format(args.cuda))

    # set seed for reproducibility
    set_seed_everywhere(args.seed, args.cuda)

    # handle dirs
    handle_dirs(args.save_dir)

    imdb_vectorizer = partial(vectorizer_from_dataframe, cls=EmbeddingVectorizer,
                              data_name='review', target_name='sentiment')

    if args.reload_from_files:
        # training from a checkpoint
        dataset = ImdbDataset.load_dataset_and_load_vectorizer(args.data_csv,
                                                               args.vectorizer_file)
    else:
        # create dataset and vectorizer
        dataset = ImdbDataset.load_dataset_and_make_vectorizer(
            args.data_csv, imdb_vectorizer)
        dataset.save_vectorizer(args.vectorizer_file)
    vectorizer = dataset.get_vectorizer()

    # Use GloVe or randomly initialized embeddings
    if args.use_glove:
        if args.reload_from_files:
            embeddings = pickle_load(args.embeddings_file)
        else:
            words = vectorizer.data_vocab._token_to_idx.keys()
            embeddings = make_embedding_matrix(path=args.glove_filepath,
                                               words=words)
            pickle_dump(embeddings, args.embeddings_file)
        print("Using pre-trained embeddings")
    else:
        print("Not using pre-trained embeddings")
        embeddings = None

    classifier = LSTM(embedding_dim=args.embedding_dim,
                      num_embeddings=len(vectorizer.data_vocab),
                      hidden_dim=args.hidden_dim,
                      output_dim=1,
                      num_layers=args.num_layers,
                      dropout=args.dropout,
                      pretrained_embeddings=embeddings,
                      padding_idx=0)

    classifier = classifier.to(args.device)

    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min', factor=0.5,
                                                     patience=1)

    # logging.basicConfig(level=logging.INFO,
    #                     format='%(levelname)-8s %(message)s',
    #                     filename=args.log_file)
    # logger = logging.getLogger()
    # logger.handlers = [logging.StreamHandler(
    #     sys.stderr), logging.FileHandler(args.log_file)]

    # run_experiment(args, classifier, loss_func,
    #                optimizer, scheduler, dataset, logger)
