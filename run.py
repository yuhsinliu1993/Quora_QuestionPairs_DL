import numpy as np
import spacy
import tensorflow as tf
import sys
import os
import argparse

from keras.callbacks import ModelCheckpoint, TensorBoard

from utils import load_glove_embeddings, to_categorical, convert_questions_to_word_ids
from input_handler import get_input_from_csv
from ESIM import ESIM


tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = None


def do_eval_and_pred(test_data_path):
    if FLAGS.load_model is None:
        raise ValueError("You need to specify the model location by --load_model=[location]")

    # Load Testing Data
    question_1, question_2, labels = get_input_from_csv(test_data_path)

    # Load Pre-trained Model
    if FLAGS.best_glove:
        import en_core_web_md
        nlp = en_core_web_md.load()  # load best-matching version for Glove
    else:
        nlp = spacy.load('en')
    embedding_matrix = load_glove_embeddings(nlp.vocab, n_unknown=FLAGS.num_unknown)  # shape=(1071074, 300)

    tf.logging.info('Build model ...')
    esim = ESIM(embedding_matrix, FLAGS.max_length, FLAGS.num_hidden, FLAGS.num_classes, FLAGS.keep_prob, FLAGS.learning_rate)

    if FLAGS.load_model:
        model = esim.build_model(FLAGS.load_model)
    else:
        raise ValueError("You need to specify the model location by --load_model=[location]")

    # Convert the "raw data" to word-ids format && convert "labels" to one-hot vectors
    q1_test, q2_test = convert_questions_to_word_ids(question_1, question_2, nlp, max_length=FLAGS.max_length, tree_truncate=FLAGS.tree_truncate)
    labels = to_categorical(np.asarray(labels, dtype='int32'))

    predictions = model.predict([q1_test, q2_test])
    print("[*] Predictions Results: \n", predictions[0])

    scores = model.evaluate([q1_test, q2_test], labels, batch_size=FLAGS.batch_size, verbose=1)
    print("========================================")
    print("[*] LOSS OF TEST DATA: %.4f" % scores[0])
    print("[*] ACCURACY OF TEST DATA: %.4f" % scores[1])


def train(train_data, val_data, batch_size, n_epochs, save_dir=None):
    # Stage 1: Read training data (csv) && Preprocessing them
    tf.logging.info('Loading training and validataion data ...')
    train_question_1, train_question_2, train_labels = get_input_from_csv(train_data)
    # val_question_1, val_question_2, val_labels = get_input_from_csv(val_data)

    # Stage 2: Load Pre-trained embedding matrix (Using GLOVE here)
    tf.logging.info('Loading pre-trained embedding matrix ...')
    if FLAGS.best_glove:
        import en_core_web_md
        nlp = en_core_web_md.load()  # load best-matching version for Glove
    else:
        nlp = spacy.load('en')
    embedding_matrix = load_glove_embeddings(nlp.vocab, n_unknown=FLAGS.num_unknown)  # shape=(1071074, 300)

    # Stage 3: Build Model
    tf.logging.info('Build model ...')
    esim = ESIM(embedding_matrix, FLAGS.max_length, FLAGS.num_hidden, FLAGS.num_classes, FLAGS.keep_prob, FLAGS.learning_rate)

    if FLAGS.load_model:
        model = esim.build_model(FLAGS.load_model)
    else:
        model = esim.build_model()

    # Stage 4: Convert the "raw data" to word-ids format && convert "labels" to one-hot vectors
    tf.logging.info('Converting questions into ids ...')
    q1_train, q2_train = convert_questions_to_word_ids(train_question_1, train_question_2, nlp, max_length=FLAGS.max_length, tree_truncate=FLAGS.tree_truncate)
    train_labels = to_categorical(np.asarray(train_labels, dtype='int32'))

    # q1_val, q2_val = convert_questions_to_word_ids(val_question_1, val_question_2, nlp, max_length=FLAGS.max_length, tree_truncate=FLAGS.tree_truncate)
    # val_labels = to_categorical(np.asarray(val_labels, dtype='int32'))

    # Stage 5: Training
    tf.logging.info('Start training ...')

    callbacks = []
    save_dir = save_dir if save_dir is not None else 'checkpoints'
    filepath = os.path.join(save_dir, "weights-{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks.append(checkpoint)

    if FLAGS.tensorboard:
        graph_dir = os.path.join('.', 'GRAPHs')
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        tb = TensorBoard(log_dir=graph_dir, histogram_freq=0, write_graph=True, write_images=True)
        callbacks.append(tb)

    model.fit(
        x=[q1_train, q2_train],
        y=train_labels,
        batch_size=batch_size,
        epochs=n_epochs,
        # validation_data=([q1_val, q2_val], val_labels),
        validation_split=0.2,
        callbacks=callbacks,
        shuffle=True,
        verbose=FLAGS.verbose
    )


def run(_):
    if FLAGS.mode == 'train':
        train(FLAGS.input_data, FLAGS.val_data, FLAGS.batch_size, FLAGS.num_epochs)
    elif FLAGS.mode == 'eval':
        do_eval_and_pred(FLAGS.test_data)
    else:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Specify number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Specify number of batch size'
    )
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=300,
        help='Specify embedding size'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Specify the max length of input sentence'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=10,
        help='Specify seed for randomization'
    )
    parser.add_argument(
        '--input_data',
        type=str,
        default="./data/processed_data/train_split.csv",
        help='Specify the location of input data',
    )
    parser.add_argument(
        '--test_data',
        type=str,
        default="./data/processed_data/test_final.csv",
        help='Specify the location of test data',
    )
    parser.add_argument(
        '--val_data',
        type=str,
        default="./data/processed_data/val_split.csv",
        help='Specify the location of test data',
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help='Specify the number of classes'
    )
    parser.add_argument(
        '--num_hidden',
        type=int,
        default=100,
        help='Specify the number of hidden units in each rnn cell'
    )
    parser.add_argument(
        '--num_unknown',
        type=int,
        default=100,
        help='Specify the number of unknown words for putting in the embedding matrix'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=4e-4,
        help='Specify dropout rate'
    )
    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.8,
        help='Specify the rate (between 0 and 1) of the units that will keep during training'
    )
    parser.add_argument(
        '--best_glove',
        action='store_true',
        help='Glove: using light version or best-matching version',
    )
    parser.add_argument(
        '--tree_truncate',
        action='store_true',
        help='Specify whether do tree_truncate or not',
        default=False
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose on training',
        default=False
    )
    parser.add_argument(
        '--load_model',
        type=str,
        help='Locate the path of the model',
    )
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Whether use tensorboard or not',
        default=True
    )
    parser.add_argument(
        '--mode',
        type=str,
        help='Specify mode: train or eval',
        required=True
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)
