# Sparse coding baseline model
import csv
import datetime
import pickle
import string 

import numpy as np
import pandas as pd
from sklearn.decomposition import SparseCoder, DictionaryLearning
from nltk.corpus import stopwords


def pretty_tdelta(tdelta):
    hours, rem = divmod(tdelta.seconds, 3600)
    mins, secs = divmod(rem, 60)
    return "{:2d} hours {:2d} mins {:2d} secs".format(hours, mins, secs)

def get_only_alpha(df):
    # Create a copy of df that is only the rows that are words (no numbers, punctuation)
    # Code is hacky, pandas is the worst
    index_df = pd.DataFrame(data=df.index)
    index_df.columns = ['words']
    filt = index_df['words'].str.contains("^[a-zA-Z]+$").values
    return df[filt]

def sample_from_dataframe(df, include_list, n=3000):
    # Take n words from the dataframe. Additionally, add the words in include_list
    sample = df.sample(n=n)
    print(sample.shape)
    for word in include_list:
        if word not in sample:
            print("adding " + word + " to dataframe")
            sample = sample.append(df.loc[word])
    return sample


def load_word_embeddings(path='glove.6B.50d.txt'):
    df = pd.read_csv(path, sep=" ", header=None, index_col=0,
                     quoting=csv.QUOTE_NONE)
    df.drop(df.index[14798], inplace=True) # Studid word is called "nan", causes bugs
    return df

def save_dict(model, path="dictionary.p"):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    np.save(path+"_basis_vectors.npy", model.components_)

def load_dict(path="dictionary.p"):
    with open(path, 'rb') as f:
        return pickle.load(f)

def learn_dictionary(X):
    dictionary = DictionaryLearning(n_components=200, fit_algorithm='lars',
                                    transform_n_nonzero_coefs=3, verbose=2)
    t0 = datetime.datetime.now()
    dictionary.fit(X)
    t1 = datetime.datetime.now()
    print("Dictionary learning took " + pretty_tdelta(t1-t0))
    return dictionary


if __name__ == "__main__":
    df = load_word_embeddings()
    filtered = get_only_alpha(df)
    print(filtered.shape)
    selected_df = sample_from_dataframe(filtered, include_list=["tie", "star", "bank", "cut", "bass", "connecticut"])
    print(selected_df.shape)
    # Save the dataframe for Ruochi:
    selected_df.to_pickle("selected_df.p")
    selected_df.to_csv("selected_df.csv", header=False)

    X = selected_df.as_matrix()
    dictionary = learn_dictionary(X)
    save_dict(dictionary, "dictionary_3000_n200_k3.p")

    dictionary = load_dict("dictionary_3000_n200_k3.p")
    basis_vectors = dictionary.components_
    print(basis_vectors.shape)
