from abc import ABC
import numpy as np
import string
from collections import defaultdict
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.corpus import lin_thesaurus


def load_vocab_embeddings(path, vocab, emb_dim=300):
    emb_mat = np.zeros((emb_dim, len(vocab)))
    with open(path, 'r') as f:
        for line in f:
            row = line.strip().split(' ')
            word = row[0]
            i = vocab.stoi[word]
            if i == 0: continue
            emb_mat[:,i] = np.array(row[1:]).astype(np.float)
        return emb_mat


class LexSubBase(ABC):
    def __init__(self, vocab):
        self.vocab = vocab
        stop_words = stopwords.words('english')
        punkt = string.punctuation
        self.spec_words = set()
        self._add_spec_words(stop_words)
        self._add_spec_words(punkt)

    def _add_spec_words(self, words):
        for el in words:
            idx = self.vocab.stoi[el]
            if idx > 0:
                self.spec_words.add(idx)

    @abstractmethod
    def get_candidates(self):
        pass


class LexSub(LexSubBase):
    "Find word substitutions for a word in context using word2vec skip-gram embedding"
    def __init__(self, vocab,
                 vector_path='.vector_cache/counter-fitted-vectors.txt'):
        super().__init__(vocab)
        self.emb_mat = load_vocab_embeddings(vector_path, vocab)

        c_ = -2*np.dot(self.emb_mat.T, self.emb_mat)
        a = np.sum(np.square(self.emb_mat), axis=0).reshape((1,-1))
        b = a.T
        self.dist_mat = a+b+c_

    def get_candidates(self, words, n_substitutes=10,
                       n_candidates=10, sbs=False,
                       **kwargs):
        
        cand_list = []
        for i, word in enumerate(words):
            cands = self._get_candidates(
                        target=word,
                        target_index=i,
                        n_candidates=n_candidates,
                        sentence=sentence
                     )
            if sbs:
                cands = self.sort_by_substitutability(cands, word, i, words,
                                                      n_substitutes)
            else:
                cands = cands[:n_substitutes]
            cand_list.append(cands)

        return cand_list

    def sort_by_substitutability(self, cands, target, target_index,
                                 sentence, n_substitutes):
        C = [c for c in sentence if c not in self.spec_words and c != target]
        scores = [self.get_substitutability(target, target_index, cand, C)
                  for cand in cands]
        sorted_cands = sorted(zip(cands, scores), key = lambda x : x[1])
        return [sub for sub, _ in sorted_cands][:n_substitutes]

    def get_substitutability(self, t, ti, s, C):
        """
        t = target word
        ti = target index
        s = candidate substitution 
        C = list of context words 
        """
        tscore = self.dist_mat[t][s]
        
        if len(C) == 0:
            cscore = 0
        else:
            cscores = [self.dist_mat[t][c] for c in C ]
            cscore = sum(cscores) / (len(C))

        return tscore + cscore

    @abstractmethod
    def _get_candidates(self, **kwargs):
        pass


class SynonymModel(LexSub):
    def _get_candidates(self, target, n_candidates=10, **kwargs):
        if target == 0: return []
        return np.argsort(self.dist_mat[target,:])[1:1+n_candidates]


class WordnetModel(LexSub):
    def __init__(self, vocab,
                 vector_path='.vector_cache/counter-fitted-vectors.txt'):
        super().__init__(vocab, vector_path)
        self.tagger = WordNetTagger()

    def get_candidates(self, **kwargs):
        self.tagger.tag(kwargs['sentence'])
        return super().get_candidates(**kwargs)
    
    def _get_candidates(self, target, target_index, n_candidates, **kwargs):
        if target == 0: return []
        tag = self.tagger.get_tag(target_index)
        word = self.vocab.itos[target]
        syns = WordnetModel.wordnet_synonyms(word, tag)
        cands = set()
        for syn in syns:
            if syn != word:
                id = self.vocab.stoi[syn]
                if id != 0:
                    cands.add(id)
        return list(cands)[:n_candidates]

    @staticmethod
    def wordnet_synonyms(word, pos_tag):
        synset = wordnet.synsets(word, pos_tag)
        return [lemma.name() for s in synset for lemma in s.lemmas()]


class LinModel(LexSub):
    def __init__(self, vocab,
                 vector_path='.vector_cache/counter-fitted-vectors.txt'):
        super().__init__(vocab, vector_path)
        self.tagger = WordNetTagger()

    def get_candidates(self, **kwargs):
        self.tagger.tag(kwargs['sentence'])
        return super().get_candidates(**kwargs)
    
    def _get_candidates(self, target, target_index, n_candidates, **kwargs):
        if target == 0: return []
        tag = self.tagger.get_tag(target_index)
        word = self.vocab.itos[target]
        syns = LinModel.lin_synonyms(word, tag)
        cands = []
        for syn in syns:
            if syn != word:
                id = self.vocab.stoi[syn]
                if id != 0:
                    print(syn)
                    cands.append(id)
        return list(cands)[:n_candidates]

    @staticmethod
    def lin_synonyms(word, pos):
        fileid = 'sim%s.lsp' % pos.upper()
        thes_entry = lin_thesaurus.scored_synonyms(word, fileid=fileid)
        thes_entry = sorted(thes_entry, key = (lambda x : x[1]), reverse = True)
        return [syn for syn, score in thes_entry]


class WordNetTagger():
    def __init__(self):
        self.tag_map = defaultdict(
            lambda: None,
            {'NN':wordnet.NOUN, 'JJ':wordnet.ADJ,
             'VB':wordnet.VERB, 'RB':wordnet.ADV}
        )
      
    def tag(self, sentence):
        self.tags = pos_tag(sentence)

    def get_tag(self, index):
        return self.tag_map[self.tags[index][1][:2]]



  
