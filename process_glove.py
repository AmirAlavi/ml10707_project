import sys

import pandas as pd
import numpy as np


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: process_glove.py <glove_embed_filename>")
        sys.exit(1)
    in_fname = sys.argv[1]
    embed_df = pd.read_csv(in_fname, delim_whitespace=True, engine='python', header=None)
    print("Embedding dimensionality: " + str(embed_df.shape[1]-1))
    embed_mat = embed_df.as_matrix(columns=range(1, embed_df.shape[1])).astype(np.float32)
    print("Embedding matrix shape: " + str(embed_mat.shape))
    vocab = embed_df.as_matrix(columns=[0]).flatten()
    np.save(in_fname + ".embed_mat.npy", embed_mat)
    np.save(in_fname + ".vocab.npy", vocab)
