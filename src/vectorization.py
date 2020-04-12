import string
from collections import Counter

from data_utils import Vocabulary


class EmbeddingVectorizer(object):
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

    @classmethod
    def from_dataframe(cls, df, cutoff=25):
        """Instantiate the vectorizer from the dataset dataframe
        
        Args:
            df (pandas.DataFrame): the target dataset
            cutoff (int): frequency threshold for including in Vocabulary 
        Returns:
            an instance of the EmbeddingVectorizer
        """
        vocab = Vocabulary()        
        for category in sorted(set(news_df.category)):
            category_vocab.add_token(category)

        word_counts = Counter()
        for title in news_df.title:
            for token in title.split(" "):
                if token not in string.punctuation:
                    word_counts[token] += 1
        
        title_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                title_vocab.add_token(word)
        
        return cls(title_vocab, category_vocab)

    @classmethod
    def from_serializable(cls, contents):
        title_vocab = \
            SequenceVocabulary.from_serializable(contents['title_vocab'])
        category_vocab =  \
            Vocabulary.from_serializable(contents['category_vocab'])

        return cls(title_vocab=title_vocab, category_vocab=category_vocab)

    def to_serializable(self):
        return {'title_vocab': self.title_vocab.to_serializable(),
                'category_vocab': self.category_vocab.to_serializable()}



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

    @classmethod
    def from_serializable(cls, contents):
        """Instantiate a OneHotVectorizer from a serializable dictionary
        
        Args:
            contents (dict): the serializable dictionary
        Returns:
            an instance of the OneHotVectorizer class
        """
        data_vocab = Vocabulary.from_serializable(contents['data_vocab'])
        data_vocab =  Vocabulary.from_serializable(contents['label_vocab'])

        return cls(data_vocab=data_vocab, label_vocab=label_vocab)

    def to_serializable(self):
        """Create the serializable dictionary for caching
        
        Returns:
            contents (dict): the serializable dictionary
        """
        return {'data_vocab': self.data_vocab.to_serializable(),
                'label_vocab': self.label_vocab.to_serializable()}