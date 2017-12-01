import matplotlib
matplotlib.use('Agg') # For running on headless server
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Masking, Bidirectional, LSTM, Dense

def get_as_one_hot(y, num_classes):
    return np.eye(num_classes)[y]

def get_data(mask_timestep=20):
    # Must generate samples, i.e. 41-grams.
    # Each gram is a wordID and the wordID at mask_timestep must be set to 0
    # Random data for now
    X_train = np.random.randint(low=1, high=400001, size=(40000, 41))
    X_valid = np.random.randint(low=1, high=400001, size=(10000, 41))
    X_test = np.random.randint(low=1, high=400001, size=(10000, 41))
    y_train = X_train[:, mask_timestep].flatten()
    y_valid = X_valid[:, mask_timestep].flatten()
    y_test = X_test[:, mask_timestep].flatten()
    y_train = get_as_one_hot(y_train, 400001)
    y_valid = get_as_one_hot(y_valid, 400001)
    y_test = get_as_one_hot(y_test, 400001)
    X_train[:, mask_timestep] = 0
    X_valid[:, mask_timestep] = 0
    X_test[:, mask_timestep] = 0
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def get_embed_weights(filename='glove.6B.50d.txt.embed_mat.npy'):
    embed_mat = np.load(filename)
    mask_embed = np.zeros(shape=(1, embed_mat.shape[1]), dtype=np.float32)
    return np.concatenate((mask_embed, embed_mat), axis=0)

def get_model(in_seq_len=41, pretrained_embed_weights=None, lstm_size=64,
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

def plot_history(history, out="train_hist.png"):
    min_train_epoch = np.argmin(history.history['loss']) + 1
    min_valid_epoch = np.argmin(history.history['val_loss']) + 1
    print("Minimum train loss achieved at epoch: " + str(min_train_epoch))
    print("Minimum valid loss achieved at epoch: " + str(min_valid_epoch))
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(out)
    plt.close()
    
    
if __name__ == "__main__":
    pt_embed_weights = get_embed_weights()
    print(pt_embed_weights.shape)
    model = get_model(pretrained_embed_weights=pt_embed_weights)
    model.summary()
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data()
    history = model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=2, validation_data=(X_valid, y_valid))
    
