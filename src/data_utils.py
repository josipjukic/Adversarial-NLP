from argparse import Namespace
from collections import Counter
from abc import (ABC, abstractmethod)
import pickle
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (Dataset, DataLoader)
from torchtext import data


def save_data(data, filepath, label='label', tokenized=True):
  entries = []
  textify = lambda x: x if tokenized else lambda x: ''.join(x)
  for example in data.examples:
    entry = dict(text=textify(example.text), label=example.label)
    entries.append(json.dumps(entry))

  json_dicts = '\n'.join(entries)
  with open(filepath, 'w') as f:
    f.write(json_dicts)
  print(f'Saved data at {filepath}.')


def save_dataset(dataset, path, label='label', tokenized=True):
  for mode in ['train', 'test', 'valid']:
    filepath = os.path.join(path, f'{mode}.json')
    save_data(dataset[mode], filepath, label, tokenized)


def load_dataset(path, include_lengths=True, lower=False, stop_words=None):
  text_field = data.Field(include_lengths=include_lengths,
                          lower=lower,
                          stop_words=stop_words)
  label_field = data.LabelField(dtype=torch.float)

  fields = {'text': ('text', text_field), 'label': ('label', label_field)}

  train_data, valid_data, test_data = data.TabularDataset.splits(
                                          path=path,
                                          train='train.json',
                                          validation='valid.json',
                                          test='test.json',
                                          format='json',
                                          fields=fields)

  return train_data, valid_data, test_data, text_field, label_field


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
