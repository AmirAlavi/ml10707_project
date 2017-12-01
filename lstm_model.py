import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Masking, Bidirectional, LSTM, Dense


def get_data(mask_timestep=20):
    # Must generate samples, i.e. 41-grams.
    # Each gram is a wordID and the wordID at mask_timestep must be set to 0
    pass

def get_embed_weights(filename='glove.6B.50d.txt.embed_mat.npy'):
    embed_mat = np.load(filename)
    mask_embed = np.zeros(shape=(1, embed_mat.shape[1]), dtype=np.float32)
    return np.concatenate((mask_embed, embed_mat), axis=0)

def get_model(in_seq_len=41, pretrained_embed_weights=None, lstm_size=128,
              lstm_act='tanh'):
    if pretrained_embed_weights is not None:
        # vocab_size should be 400000 + 1 (index 0 returns all zeros, for masking)
        vocab_size = pretrained_embed_weights.shape[0]
        embed_size = pretrained_embed_weights.shape[1]
    else:
        raise ValueError("Must provide pretrained embedding weights")
    
    model = Sequential()
    model.add(
        Embedding(vocab_size, embed_size, input_length=in_seq_len,
                  mask_zero=True, weights=[pretrained_embed_weights],
                  trainable=False)
    )
    model.add(
        Bidirectional(
            LSTM(lstm_size, activation=lstm_act, )
        )
    )
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model
    
    
if __name__ == "__main__":
    pt_embed_weights = get_embed_weights()
    print(pt_embed_weights.shape)
    model = get_model(pretrained_embed_weights=pt_embed_weights)
    model.summary()
    # X_train, y_train, X_valid, y_valid = get_data()
