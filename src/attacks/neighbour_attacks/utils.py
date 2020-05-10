import numpy as np


def softmax(x):
    exp = np.exp(x)
    return exp / exp.sum()


def prob_normalize(x):
    s = np.sum(x)
    if s == 0: return x
    return x / s