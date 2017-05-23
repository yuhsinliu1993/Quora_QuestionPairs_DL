# Kaggle Competition: Quora Question Pairs Problem

See more infomation on [https://www.kaggle.com/c/quora-question-pairs](https://www.kaggle.com/c/quora-question-pairs)

Reference
---------

1. Code is based on the paper "A decomposable attention model for natural language inference (2016)" proposed by Aparikh, Oscart, Dipanjand, Uszkoreit. See more detail on [https://arxiv.org/abs/1606.01933](https://arxiv.org/abs/1606.01933)

2. "Reasoning about entailment with neural attention (2016)" proposed by Tim Rockta schel. See more detail on [https://arxiv.org/abs/1509.06664](https://arxiv.org/abs/1509.06664)

3. "Neural Machine Translation by Jointly Learning to Align and Translate (2016)" proposed by Yoshua Bengio, Dzmitry Bahdanau, KyungHyun Cho. See more detail on [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)


Prerequisites
-------------

- python 2.7
- numpy
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://github.com/fchollet/keras) (need to install from source, not from pip)
- [spaCy](https://spacy.io)

Download spaCy pre-trained word2vec (Glove)

    # out-of-the-box: download best-matching default model
    $ python -m spacy download en

    # download best-matching version of specific model for your spaCy installation
    $ python -m spacy download en_core_web_md


Usage
-----

To train a model on default settings: (epochs: 10, embedding size: 128, hidden units: 100, learning rate: 0.001, input data: ./data/train.csv)

    $ python run.py --mode=train --verbose

To test a model:

    $ python run.py --mode=eval


All training option:
```
usage: run.py [-h] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
              [--embedding_size EMBEDDING_SIZE] [--max_length MAX_LENGTH]
              [--seed SEED] [--input_data INPUT_DATA] [--test_data TEST_DATA]
              [--num_classes NUM_CLASSES] [--num_hidden NUM_HIDDEN]
              [--num_unknown NUM_UNKNOWN] [--learning_rate LEARNING_RATE]
              [--keep_prob KEEP_PROB] --mode MODE [--best_glove] [--encode]
              [--tree_truncate] [--verbose]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Specify number of epochs
  --batch_size BATCH_SIZE
                        Specify number of batch size
  --embedding_size EMBEDDING_SIZE
                        Specify embedding size
  --max_length MAX_LENGTH
                        Specify the max length of input sentence
  --seed SEED           Specify seed for randomization
  --input_data INPUT_DATA
                        Specify the location of input data
  --test_data TEST_DATA
                        Specify the location of test data
  --num_classes NUM_CLASSES
                        Specify the number of classes
  --num_hidden NUM_HIDDEN
                        Specify the number of hidden units in each rnn cell
  --num_unknown NUM_UNKNOWN
                        Specify the number of unknown words for putting in the
                        embedding matrix
  --learning_rate LEARNING_RATE
                        Specify dropout rate
  --keep_prob KEEP_PROB
                        Specify the rate (between 0 and 1) of the units that
                        will keep during training
  --mode MODE           Specify mode: train or eval
  --best_glove          Glove: using light version or best-matching version
  --encode              If encode is assigned, sentence will pass through
                        BiRNN-Encoding Layer
  --tree_truncate       Specify whether do tree_truncate or not
  --verbose             Verbose on training
```
