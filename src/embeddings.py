import numpy as np


GLOVE_6B_50D_PATH = '../embeddings/glove.6B/glove.6B.50d.txt'

def load_glove_embeddings(path):
    embeddings_dict = {}
    with open(path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(line[1:]).astype(np.float)
            embeddings_dict[word] = vector
    return embeddings_dict