from argparse import Namespace
from collections import Counter
from abc import (ABC, abstractmethod)
import pickle
import json
import os
import re
import spacy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (Dataset, DataLoader)
from torchtext import data


def save_data(data, filepath, nlp, id):
    entries = []
    for example in data.examples:
        entry = dict(text=[token.text for token in nlp(example.text)],
                     label=example.label,
                     raw=example.text,
                     id=id)
        entries.append(json.dumps(entry))
        id += 1

    json_dicts = '\n'.join(entries)
    with open(filepath, 'w') as f:
        f.write(json_dicts)
    print(f'Saved data at {filepath}.')
    return id


def save_dataset(dataset, path):
    nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
    id = 0
    for mode in ['train', 'test', 'valid']:
        filepath = os.path.join(path, f'{mode}.json')
        id = save_data(dataset[mode], filepath, nlp, id)


def load_dataset(path, include_lengths=True, lower=False, stop_words=None):
    TEXT = data.Field(include_lengths=include_lengths,
                      lower=lower,
                      stop_words=stop_words)
    LABEL = data.LabelField(dtype=torch.float)
    RAW = data.RawField()
    ID = data.RawField()

    fields = {'text': ('text', TEXT),
              'label': ('label', LABEL),
              'raw': ('raw', RAW),
              'id': ('id', ID)}

    splits = data.TabularDataset.splits(
                                path=path,
                                train='train.json',
                                validation='valid.json',
                                test='test.json',
                                format='json',
                                fields=fields)

    return splits, (TEXT, LABEL, RAW, ID)


def spacy_revtok(nlp, tokens):
    return ''.join(token.text_with_ws for token in tokens)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def expand_paths(args):
    args.model_save_file = os.path.join(args.save_dir, args.model_save_file)
    args.train_state_file = os.path.join(args.save_dir, args.train_state_file)


def pickle_dump(obj, filepath):
    """
    A static method for saving an object in pickle format.

    Arguments
    ---------
    obj
        Object to be serialized.
    filepath : str
        The location of the serialized object.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(filepath):
    """
    A static method for loading an object from pickle file.

    Arguments
    ---------
    filepath : str
        The location of the serialized object.

    Returns
    -------
        An instance of a the serialized object.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def json_dump(obj, filepath):
    """
    A static method for saving an object in json format.

    Arguments
    ---------
    obj
        Object to be serialized.
    filepath : str
        The location of the serialized object.
    """
    with open(filepath, 'w') as f:
        json.dump(obj, f)


def json_load(filepath):
    """
    A static method for loading an object from json file.

    Arguments
    ---------
    filepath : str
        The location of the serialized object.

    Returns
    -------
        An instance of a the serialized object.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


# logging.basicConfig(level=logging.INFO,
#                     format='%(levelname)-8s %(message)s',
#                     filename=args.log_file)
# logger = logging.getLogger()
# logger.handlers = [logging.StreamHandler(
#     sys.stderr), logging.FileHandler(args.log_file)]
