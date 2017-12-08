# Sparse coding baseline model
import csv
import datetime
import pickle
import string 

import numpy as np
import pandas as pd
from sklearn.decomposition import SparseCoder, DictionaryLearning
from scipy.spatial.distance import cdist


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
    dictionary = DictionaryLearning(n_components=500, fit_algorithm='lars',
                                    transform_n_nonzero_coefs=5, verbose=2)
    t0 = datetime.datetime.now()
    dictionary.fit(X)
    t1 = datetime.datetime.now()
    print("Dictionary learning took " + pretty_tdelta(t1-t0))
    return dictionary

def find_nearest_words(atoms_of_discourse, embeddings):
    for i, atom in enumerate(atoms_of_discourse):
        print("\tAtom " + str(i+1))
        # Compute distances to all word embeddings
        dist_mat = cdist(atom.reshape(1,-1), embeddings, metric='cosine')
        sorted_idx = np.argsort(dist_mat, axis=None)
        top_10 = embeddings.index[sorted_idx][:10].values
        print("\t\t" + str(top_10))
        

def get_atoms(model, word_embedding):
    sparse_code = model.transform(word_embedding.reshape(1, -1)).flatten()
    activated_atoms_filter = sparse_code > 0
    # report the atom numbers (which of the 500)
    print("\tAtoms: " + str(np.arange(len(sparse_code))[activated_atoms_filter]))
    activated_atoms = model.components_[activated_atoms_filter]
    return activated_atoms

def analyze_atoms_of_discourse(model_file, embeddings_file, words=["tie", "star", "bank", "cut", "bass", "connecticut"]):
    model = load_dict(model_file)
    word_embeddings = pd.read_pickle(embeddings_file)
    for word in words:
        print("Looking at word: ", word)
        word_embedding = word_embeddings.loc[word].values
        atoms = get_atoms(model, word_embedding)
        find_nearest_words(atoms, word_embeddings)
        


if __name__ == "__main__":
    analyze_atoms_of_discourse("results/glove_10000_n500_k3/dictionary_10000_n500_k3.p", "results/glove_10000_n500_k3/selected_df_10k.p")


# if __name__ == "__main__":
#     df = load_word_embeddings()
#     filtered = get_only_alpha(df)
#     print(filtered.shape)
#     selected_df = sample_from_dataframe(filtered, include_list=["tie", "star", "bank", "cut", "bass", "connecticut"], n=10000)
#     print(selected_df.shape)
#     # Save the dataframe for Ruochi:
#     selected_df.to_pickle("selected_df_10k_k5.p")
#     selected_df.to_csv("selected_df_10k_k5.csv", header=False)

#     X = selected_df.as_matrix()
#     dictionary = learn_dictionary(X)
#     save_dict(dictionary, "dictionary_10000_n500_k5.p")

#     dictionary = load_dict("dictionary_10000_n500_k5.p")
#     basis_vectors = dictionary.components_
#     print(basis_vectors.shape)

#     # 3000 words, 200 components, k=3, Dictionary learning took  0 hours 54 mins 50 secs, 1000 iterations, final cost: 22191.404
#     # 4000 words, 200 components, k=3, Dictionary learning took  1 hours 18 mins 23 secs, 1000 iterations, final cost: 30077.670
#     # 5000 words, 500 components, k=3, Dictionary learning took  2 hours 12 mins 44 secs, 1000 iterations, final cost: 34718.476
#     # 10000 words, 500 components, k=3, Dictionary learning took  5 hours  0 mins 56 secs, 1000 iterations, final cost: 71434.546
#     # 10000 words, 500 components, k=5, Dictionary learning took  5 hours  8 mins 57 secs, 1000 iterations, final cost: 71518.951
