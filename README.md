# 10-707 Project (Word Sense Induction using LSTMs)

## Constructing the training dataset
Jupyter Notebook file: construct_dataset.ipynb

* Reads in Glove vectors and constructs a vocabulary from it so that order remains the same between GloVe and vocab used.
* Adds a "MASK" token at beginning of vocabulary so that indexing of GloVe starts from 1 and also facilitates masking during training the LSTM
* Reads through Wikipedia dataset and extracts n-grams at a given stride
* Constructs the train, validation and test sets which are basically n-grams that are in the form of indexes into GloVe.
* Any n-gram with words not in GloVe vocabulary are removed
* Test and validation split are constructed such that the labels (middle word held out) in these sets are a subset of the labels present in training set
* Tested on Python2.7.x (Anaconda distribution)
