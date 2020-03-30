from argparse import Namespace
from collections import Counter
from abc import (ABC, abstractmethod)
import pickle
import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (Dataset, DataLoader)


class Vocabulary(object):
    """ Class to process text and extract vocabulary for mapping. """

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        Arguments
        ---------
        token_to_idx : dict
            A pre-existing map of tokens to indices.
        add_unk : bool
            A flag that indicates whether to add the UNK token.
        unk_token : str
            The UNK token to add into the Vocabulary.
        """
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token 
                              for token, idx in self._token_to_idx.items()}
        
        self._add_unk = add_unk
        self._unk_token = unk_token
        
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token) 
        
        
    def to_serializable(self):
        """ Returns a dictionary that can be serialized. """
        return {'token_to_idx': self._token_to_idx, 
                'add_unk': self._add_unk, 
                'unk_token': self._unk_token}

    @classmethod
    def from_serializable(cls, contents):
        """ Instantiates the Vocabulary from a serialized dictionary. """
        return cls(**contents)

    def add_token(self, token):
        """
        Update mapping dicts based on the token.

        Arguments
        ---------
        token : str
            The item to add into the Vocabulary.

        Returns
        -------
        index : int
            The integer corresponding to the token.
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index
    
    def add_many(self, tokens):
        """
        Add a list of tokens into the Vocabulary
        
        Arguments
        ---------
        tokens : list
            A list of string tokens.
        
        Returns
        -------
        indices : list
            A list of indices corresponding to the tokens.
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """
        Retrieve the index associated with the token 
        or the UNK index if token isn't present.
        
        Arguments
        ---------
        token : str
            The token to look up.

        Returns
        -------
        index : int
            The index corresponding to the token.

        Notes
        -----
        `unk_index` needs to be >=0 (having been added into the Vocabulary) 
        for the UNK functionality.
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """
        Return the token associated with the index
        
        Arguments
        ---------
        index : int
            The index to look up.

        Returns
        -------
        token : str
            The token corresponding to the index.

        Raises
        ------
        KeyError
            If the index is not in the Vocabulary.
        """
        if index not in self._idx_to_token:
            raise KeyError(f'The index ({index}) is not in the Vocabulary')
        return self._idx_to_token[index]

    def __str__(self):
        return f'<Vocabulary(size={len(self)})>'

    def __len__(self):
        return len(self._token_to_idx)


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


class DataManager(ABC, Dataset):
    def __init__(self, df, vectorizer):
        """
        Arguments
        ---------
        df : pandas.DataFrame
            The dataset.
        vectorizer
            Vectorizer instantiated from dataset.
        """
        self.df = df
        self._vectorizer = vectorizer

        self.train_df = self.df[self.df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.df[self.df.split=='val']
        self.validation_size = len(self.val_df)

        self.test_df = self.df[self.df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, df_csv, init_vectorizer):
        """
        Load dataset and make a new vectorizer from scratch.
        
        Arguments
        ---------
        df_csv : str
            Location of the dataset.
        init_vectorizer : method
            Method that receieves train dataframe as an argument
            and returns constructed vectorizer.

        Returns
        -------
            An instance of DataManager.
        """
        df = pd.read_csv(df_csv)
        train_df = df[df.split=='train']
        return cls(df, init_vectorizer(train_df))
    
    @classmethod
    def load_dataset_and_load_vectorizer(cls, df_csv, vectorizer_filepath):
        """
        Load dataset and the corresponding vectorizer. 
        Used in the case in the vectorizer has been cached for re-use.
        
        Arguments
        ---------
        review_csv : str
            Location of the dataset.
        vectorizer_filepath :
            Location of the saved vectorizer.

        Returns
        -------
            An instance of DataManager.
        """
        df = pd.read_csv(df_csv)
        vectorizer = pickle_load(vectorizer_filepath)
        return cls(df, vectorizer)

    def save_vectorizer(self, vectorizer_filepath):
        """
        Saves the vectorizer to disk using pickle.
        
        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        pickle_dump(self._vectorizer, vectorizer_filepath)

    def get_vectorizer(self):
        """ Returns the vectorizer. """
        return self._vectorizer

    def set_split(self, split="train"):
        """
        Selects the split in the dataset using a column in the dataframe.
        
        Arguments
        ---------
        split : str
            One of "train", "val", or "test".
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    @abstractmethod
    def __getitem__(self, index):
        """
        The primary entry point method for PyTorch datasets.
        
        Arguments
        ---------
        index : int
            The index to the data point. 
        Returns:
            A dictionary holding the data point's features (x_data) and label (y_target).
        """
        pass
        # row = self._target_df.iloc[index]

        # review_vector = \
        #     self._vectorizer.vectorize(row.review)

        # rating_index = \
        #     self._vectorizer.rating_vocab.lookup_token(row.rating)

        # return {'x_data': review_vector,
        #         'y_target': rating_index}

    def get_num_batches(self, batch_size):
        """
        Given a batch size, returns the number of batches in the dataset.
        
        Arguments
        ---------
        batch_size : int

        Returns
        -------
            Number of batches in the dataset.
        """
        return len(self) // batch_size  


def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device='cpu'):
    """
    A generator function which wraps the PyTorch DataLoader. It will 
    ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)