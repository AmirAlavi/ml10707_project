# Sparse coding baseline model
import csv
import datetime

import numpy as np
import pandas as pd
from sklearn.decomposition import SparseCoder, DictionaryLearning

def pretty_tdelta(tdelta):
    hours, rem = divmod(tdelta.seconds, 3600)
    mins, secs = divmod(rem, 60)
    return "{:2d} hours {:2d} mins {:2d} secs".format(hours, mins, secs)

def load_word_embeddings(path='glove.6B.50d.txt'):
    return pd.read_csv(path, sep=" ", header=None, index_col=0,
                       quoting=csv.QUOTE_NONE)

def learn_dictionary(X):
    dictionary = DictionaryLearning(n_components=100, fit_algorithm='lars',
                                    transform_n_nonzero_coefs=5, verbose=True)
    t0 = datetime.datetime.now()
    dictionary.fit(X)
    t1 = datetime.datetime.now()
    print("Dictionary learning took " + pretty_tdelta(t1-t0))


if __name__ == "__main__":
    dataframe = load_word_embeddings()
    X = dataframe.as_matrix()
    dictionary = learn_dictionary(X[:1000])