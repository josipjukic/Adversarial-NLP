import os
from argparse import Namespace
from functools import partial

import pandas as pd
import torch
import torch.nn as nn

from vectorization import EmbeddingVectorizer
from vectorization import vectorizer_from_dataframe
from embeddings import GLOVE_6B_100D_PATH
from embeddings import make_embedding_matrix
from data_utils import (set_seed_everywhere, handle_dirs, pickle_dump, pickle_load)
from data_utils import ImdbDataset
from models import RNN


if __name__ == '__main__':

    cwd = os.getcwd()
    root = os.path.join(cwd, '..')
    imdb_file_path = os.path.join(root, 'datasets/imdb/imdb_clean_split.csv')
    emb_file_path = os.path.join(root, GLOVE_6B_100D_PATH)
    save_path = os.path.join(root, 'save/imdb')

    args = Namespace(
        # Data and Path hyper parameters
        data_csv=imdb_file_path,
        vectorizer_file="imdb_vectorizer.pkl",
        embeddings_file="imdb_embeddings.pkl",
        model_state_file="imdb_model.pkl",
        save_dir=save_path,
        # Model hyper parameters
        glove_filepath=emb_file_path, 
        use_glove=True,
        embedding_dim=100, 
        hidden_dim=100, 
        # Training hyper parameter
        seed=42, 
        learning_rate=0.001, 
        dropout_p=0., 
        batch_size=128, 
        num_epochs=100, 
        early_stopping_criteria=5, 
        # Runtime option
        cuda=True, 
        catch_keyboard_interrupt=True, 
        reload_from_files=True,
        expand_filepaths_to_save_dir=True)

    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)
        args.embeddings_file = os.path.join(args.save_dir,
                                            args.embeddings_file)
        args.model_state_file = os.path.join(args.save_dir,
                                             args.model_state_file)
        
        print("Expanded filepaths: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.embeddings_file))
        print("\t{}".format(args.model_state_file))

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
        dataset = ImdbDataset.load_dataset_and_make_vectorizer(args.data_csv, imdb_vectorizer)
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

    classifier = RNN(embedding_dim=args.embedding_size, 
                     num_embeddings=len(vectorizer.data_vocab),
                     hidden_dim=args.hidden_dim, 
                     num_classes=len(vectorizer.label_vocab), 
                     dropout_p=args.dropout_p,
                     pretrained_embeddings=embeddings,
                     padding_idx=0)


    # classifier = classifier.to(args.device)
    # dataset.class_weights = dataset.class_weights.to(args.device)
        
    loss_func = nn.CrossEntropyLoss(dataset.class_weights)
    print(dataset.class_weights)
    # optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
    #                                         mode='min', factor=0.5,
    #                                         patience=1)