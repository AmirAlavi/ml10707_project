# Sparse coding baseline model
import csv
import datetime
import pickle

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

def save_dict(model, path="dictionary.p"):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_dict(path="dictionary.p"):
    with open(path, 'rb') as f:
        return pickle.load(f)

def learn_dictionary(X):
    dictionary = DictionaryLearning(n_components=2000, fit_algorithm='lars',
                                    transform_n_nonzero_coefs=5, verbose=2)
    t0 = datetime.datetime.now()
    dictionary.fit(X)
    t1 = datetime.datetime.now()
    print("Dictionary learning took " + pretty_tdelta(t1-t0))


if __name__ == "__main__":
    # dataframe = load_word_embeddings()
    # X = dataframe.as_matrix()
    # #dictionary = learn_dictionary(X[:1000]) # 13 mins 40 secs, m=2000, k=5
    # save_dict(dictionary, "dictionary_1k.p")
    # dictionary = learn_dictionary(X[:10000])
    # save_dict(dictionary, "dictionary_10k.p")

    dictionary = load_dict("dictionary_1k.p")
    print(type(dictionary))
    basis_vectors = dictionary.components_ # Each row is a learned basis vector
    print(basis_vectors.shape)