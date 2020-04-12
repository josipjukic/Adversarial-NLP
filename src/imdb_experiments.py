import os
from embeddings import *
import pandas as pd


cwd = os.getcwd()
dir = os.path.join(cwd, '..')
imdb_file_path = os.path.join(dir, 'datasets/IMDB.csv')
emb_file_path = os.path.join(dir, GLOVE_6B_50D_PATH)

df = pd.read_csv(imdb_file_path)
print(df.summary())