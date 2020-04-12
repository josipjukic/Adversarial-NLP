import string
from collections import Counter

from data_utils import Vocabulary   


class EmbeddingVectorizer():
    def __init__(self, data_vocab, label_vocab):
        """
        Args:
            data_vocab (Vocabulary): maps words to integers
            label_vocab (Vocabulary): maps class labels to integers
        """
        self.data_vocab = data_vocab
        self.label_vocab = label_vocab

    def vectorize(self, text, vector_length=-1):
        """
        Args:
            text (str): the string of words separated by a space
            vector_length (int): an argument for forcing the length of index vector
        Returns:
            vectorized text (numpy.array)
        """
        indices = [self.data_vocab.begin_seq_index]
        indices.extend(self.title_vocab.lookup_token(token) 
                       for token in title.split(" "))
        indices.append(self.title_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.data_vocab.mask_index

        return out_vector


class OneHotVectorizer():
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    def __init__(self, data_vocab, label_vocab):
        """
        Args:
            data_vocab (Vocabulary): maps words to integers
            label_vocab (Vocabulary): maps class labels to integers
        """
        self.data_vocab = data_vocab
        self.label_vocab = label_vocab

    def vectorize(self, text):
        """Create a collapsed one-hit vector for the review
        
        Args:
            review (str): the review 
        Returns:
            one_hot (np.ndarray): the collapsed one-hot encoding 
        """
        one_hot = np.zeros(len(self.data_vocab), dtype=np.float32)
        
        for token in text.split(' '):
            if token not in string.punctuation:
                one_hot[self.data_vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, corpus, labels, cutoff=25):
        """Instantiate the vectorizer from the dataset dataframe
        
        Args:
            df (pandas.DataFrame): the dataset
            cutoff (int): the parameter for frequency-based filtering
        Returns:
            an instance of the OneHotVectorizer
        """
        data_vocab = Vocabulary(add_unk=True)
        label_vocab = Vocabulary(add_unk=False)
        
        # Add ratings
        for label in sorted(set(labels)):
            label_vocab.add_token(label)

        # Add top words if count > provided count
        word_counts = Counter()
        for text in corpus:
            for word in text.split(' '):
                if word not in string.punctuation:
                    word_counts[word] += 1
               
        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)

        return cls(review_vocab, rating_vocab)
