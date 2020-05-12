import numpy as np
import torch


GLOVE_6B_50D_PATH = 'embeddings/glove.6B/glove.6B.50d.txt'
GLOVE_6B_100D_PATH = 'embeddings/glove.6B/glove.6B.100d.txt'


def load_glove_embeddings(path):
    word_to_index = {}
    embeddings = []
    with open(path, 'r') as f:
        for index, line in enumerate(f):
            values = line.split()
            word = values[0]
            word_to_index[word] = index
            vector = np.array(values[1:]).astype(np.float)
            embeddings.append(vector)
    return word_to_index, np.array(embeddings)


def make_embedding_matrix(path, words):
    word_to_index, embeddings = load_glove_embeddings(path)
    embedding_size = embeddings.shape[1]
    N = len(words) + 2 # unknown and mask
    final_emb = np.zeros((N, embedding_size))
    final_emb[0, :] = np.zeros(embedding_size)
    final_emb[1, :] = np.zeros(embedding_size)

    for i, word in enumerate(words, 2):
        if word in word_to_index:
            final_emb[i, :] = embeddings[word_to_index[word]]
        else:
            emb_i = np.random.randn(1, embedding_size)
            final_emb[i, :] = emb_i
    return final_emb


def get_embeddings(TEXT, args):
    pretrained_embeddings = TEXT.vocab.vectors
    pretrained_embeddings[args.UNK_IDX] = torch.zeros(args.embedding_dim)
    pretrained_embeddings[args.PAD_IDX] = torch.zeros(args.embedding_dim)
    return pretrained_embeddings
