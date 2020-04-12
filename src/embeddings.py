import numpy as np


GLOVE_6B_50D_PATH = 'embeddings/glove.6B/glove.6B.100d.txt'


def load_glove_embeddings(path):
    word_to_index = {}
    embeddings = []
    with open(path, 'r') as f:
        for index, line in enumerate(f):
            values = line.split()
            word = values[0]
            word_to_index[word] = index
            vector = np.array(line[1:]).astype(np.float)
            embeddings.append(vector)
    return word_to_index, embeddings


def make_embedding_matrix(path, words):
    word_to_index, embeddings = load_glove_embeddings(path) 
    embedding_size = embeddings.shape[1]
    final_emb = np.zeros(len(words), embedding_size)

    for i, word in enumerate(words):
        if word in word_to_index:
            final_emb[i, :] = embeddings[word_to_index[word]]
        else:
            emb_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(emb_i)
            final_emb[i, :] = emb_i
    return final_emb