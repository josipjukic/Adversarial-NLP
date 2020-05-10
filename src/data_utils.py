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
from torchtext import data


def save_data(data, filepath, tokenize, id, save_raw, save_id):
    entries = []
    for example in data.examples:
        entry = dict(text=tokenize(example.text),
                     label=example.label)
        if save_raw:
            entry['raw'] = example.text
        if save_id:
            entry['id'] = id
        entries.append(json.dumps(entry))
        id += 1

    json_dicts = '\n'.join(entries)
    with open(filepath, 'w') as f:
        f.write(json_dicts)
    print(f'Saved data at {filepath}.')
    return id


def save_nli_data(data, filepath, tokenize, id, save_raw, save_id):
    entries = []
    for example in data.examples:
        entry = dict(premise=tokenize(example.premise),
                     hypothesis=tokenize(example.hypothesis),
                     label=example.label)
        if save_raw:
            entry['raw_premise'] = example.premise
            entry['raw_hypothesis'] = example.hypothesis
        if save_id:
            entry['id'] = id
        entries.append(json.dumps(entry))
        id += 1

    json_dicts = '\n'.join(entries)
    with open(filepath, 'w') as f:
        f.write(json_dicts)
    print(f'Saved data at {filepath}.')
    return id


DATA_TYPES = {'classification': save_data, 'NLI': save_nli_data}


def save_dataset(dataset, path, tokenize=None, save_raw=True,
                 save_id=True, data_type='classification'):
    if not tokenize:
        nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        tokenize = lambda x: [token.text for token in nlp(x)]
    id = 0
    save_fn = DATA_TYPES[data_type]
    for mode in ['train', 'test', 'valid']:
        filepath = os.path.join(path, f'{mode}.json')
        id = save_fn(dataset[mode], filepath, tokenize, id, save_raw, save_id)


def load_dataset(path, include_lengths=True, lower=False, stop_words=None,
                 load_raw=False, load_id=False, float_label=True):
    TEXT = data.Field(include_lengths=include_lengths,
                      lower=lower,
                      stop_words=stop_words)
    label_type = torch.float if float_label else torch.long
    LABEL = data.LabelField(dtype=label_type)
    RAW = data.RawField()
    ID = data.RawField()

    fields = {'text': ('text', TEXT),
              'label': ('label', LABEL)}

    if load_raw:
        fields['raw'] = ('raw', RAW)
        RAW.is_target = True

    if load_id:
        fields['id'] = ('id', ID)
        ID.is_target = True

    splits = data.TabularDataset.splits(
                                path=path,
                                train='train.json',
                                validation='valid.json',
                                test='test.json',
                                format='json',
                                fields=fields)

    return splits, (TEXT, LABEL, RAW, ID)


def load_data(LOAD_PATH, include_lengths=True, lower=False,
              stop_words=None, load_raw=True, load_id=True,
              float_label=True, MAX_VOCAB_SIZE=25_000,
              EMBEDDINGS_FILE='glove.6B.100d'):

    splits, fields = load_dataset(LOAD_PATH,
                                  include_lengths,
                                  lower,
                                  stop_words,
                                  load_id,
                                  load_raw,
                                  float_label)
    # TEXT
    fields[0].build_vocab(splits[0], 
                          max_size=MAX_VOCAB_SIZE, 
                          vectors=EMBEDDINGS_FILE, 
                          unk_init=torch.Tensor.normal_)
    # LABEL
    fields[1].build_vocab(splits[0])
    return splits, fields


def load_dataset_for_transformer(path, tokenizer, lower=False,
                                 stop_words=None, load_raw=True,
                                 load_id=True, float_label=True,
                                 max_len=512):
    
    postpro = lambda xs, _: [tokenizer.convert_tokens_to_ids(x[:max_len])
                             for x in xs]

    TEXT = data.Field(use_vocab=False,
                      postprocessing=postpro,
                      pad_token=tokenizer.pad_token_id,
                      lower=lower,
                      stop_words=stop_words)
    label_type = torch.float if float_label else torch.long
    LABEL = data.LabelField(dtype=label_type)
    RAW = data.RawField()
    ID = data.RawField()

    fields = {'text': ('text', TEXT),
              'label': ('label', LABEL)}

    if load_raw:
        fields['raw'] = ('raw', RAW)
        RAW.is_target = True

    if load_id:
        fields['id'] = ('id', ID)
        ID.is_target = True

    splits = data.TabularDataset.splits(
                                path=path,
                                train='train.json',
                                validation='valid.json',
                                test='test.json',
                                format='json',
                                fields=fields)

    return splits, (TEXT, LABEL, RAW, ID)


def load_data_for_transformer(LOAD_PATH, tokenizer, lower=False,
                              stop_words=None, load_raw=True,
                              load_id=True, float_label=True,
                              max_len=512):

    splits, fields = load_dataset_for_transformer(LOAD_PATH,
                                                  tokenizer,
                                                  lower,
                                                  stop_words,
                                                  load_id,
                                                  load_raw,
                                                  float_label,
                                                  max_len)
    # LABEL
    fields[1].build_vocab(splits[0])
    return splits, fields


def load_nli_dataset(path, lower=False, stop_words=None,
                     load_raw=True, load_id=True,
                     float_label=False):
    TEXT = data.Field(lower=lower,
                      stop_words=stop_words)
    label_type = torch.float if float_label else torch.long
    LABEL = data.LabelField(dtype=label_type)
    RAW = data.RawField()
    ID = data.RawField()

    fields = {'premise': ('premise', TEXT),
              'hypothesis': ('hypothesis', TEXT),
              'label': ('label', LABEL)}

    if load_raw:
        fields['raw_premise'] = ('raw_premise', RAW)
        fields['raw_hypothesis'] = ('raw_hypothesis', RAW)

    if load_id:
        fields['id'] = ('id', ID)

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


def indexing_adversarial_text(raw, nlp, indices, transform):
    adv_words = [token.text_with_ws for token in nlp(raw)]
    for i in indices:
        if i >= len(adv_words): continue
        adv_words[i] = transform(adv_words[i])
    return ''.join(adv_words)


def replacing_adversarial_text(raw, nlp, new_text, vocab):
    tokens = nlp(raw)
    new_words = []
    for i, idx in enumerate(new_text):
        new_words.append(vocab.itos[idx] + tokens[i].whitespace_)
    return ''.join(new_words)


def reconstruct(tensor, vocab):
    words = [vocab.itos[idx] for idx in tensor]
    return ' '.join(words)


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


def set_seed_everywhere(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
