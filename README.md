# 10-707 Project (Word Sense Induction using LSTMs)

## Constructing the training dataset (Python 2.7.x)
Jupyter Notebook file: construct_dataset.ipynb

* Reads in Glove vectors and constructs a vocabulary from it so that order remains the same between GloVe and vocab used.
* Adds a "MASK" token at beginning of vocabulary so that indexing of GloVe starts from 1 and also facilitates masking during training the LSTM
* Reads through Wikipedia dataset (example Wikipedia subset provided in wikipedia_sentences_test.txt) and extracts n-grams at a given stride
* Constructs the train, validation and test sets which are basically n-grams that are in the form of indexes into GloVe.
* Any n-gram with words not in GloVe vocabulary are removed
* Test and validation split are constructed such that the labels (middle word held out) in these sets are a subset of the labels present in training set
* Tested on Python2.7.x (Anaconda distribution)

## LSTM Language Model Script: lmo (Python 3.x)
```
Lstm language model (lmo), trained model used for Word Sense Induction downstream
Usage:
    lmo train <input_path> [--out_path=<path> --size=<size> --activation=<act> --bidirectional --epochs=<n_epochs> --batch_size=<n_batch> --gpus=<n_gpus> --train_max_samples=<n> --valid_max_samples=<n>]
    lmo eval  <model_file> <input_file> <output_file>
    lmo (-h | --help)
Options:
    -h --help                             Show this screen.
    -o <path>, --out_path=<path>          Specify output directory where command output is saved (i.e. plots, models, matrices). If not specified,
                                          defaults to a current date-time stamp. [default: CurTimeStamp]
    -s <size>, --size=<size>              Size of the LSTM layer. [default: 128]
    -a <act>, --activation=<act>          Activation function for LSTM layer. [default: tanh]
    -B --bidirectional                    Use a bidirectional LSTM.
    -e <n_epochs>, --epochs=<n_epochs>    Specify number of epochs to train for. [default: 100]
    -b <n_batch>, --batch_size=<n_batch>  Specify mini-batch size. [default: 512]
    -g <n_gpus>, --gpus=<n_gpus>          Specify number of gpu cards to use (Data parallelism if > 1). [default: 1]
    -t <n>, --train_max_samples=<n>       Only use n (randomly selected) samples from the training set. 0 means use all samples. [default: 0]
    -v <n>, --valid_max_samples=<n>       Only use n (randomly selected) samples from the validation set. 0 means use all samples. [default: 0]
```

## Sparse Coding (Python 3.x)
Reads in GloVe vectors and uses scikitlearn to do sparse coding and learn atoms of discourses similar to the method described in Arora, Sanjeev, et al. "Linear algebraic structure of word senses, with applications to polysemy." arXiv preprint arXiv:1601.03764 (2016).
