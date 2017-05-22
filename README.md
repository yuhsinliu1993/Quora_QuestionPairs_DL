# Kaggle Competition: Quora Question Pairs Problem
	Code is based on the paper "A decomposable attention model for natural language inference" proposed by aparikh, oscart, dipanjand, uszkoreit (Google)
	See more detail on https://arxiv.org/abs/1606.01933


## Usage ##
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
