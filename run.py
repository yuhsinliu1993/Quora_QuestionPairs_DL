import numpy as np
import spacy
import tensorflow as tf
import sys
import os
import argparse

from utils import load_glove_embeddings, to_categorical, convert_questions_to_word_ids
from input_handler import get_input_from_csv

from models import EmbeddingLayer, BiLSTM_Layer, Composition_Layer, Pooling_Layer, attention, attention_output, attention_softmax3d, attention_softmax3d_output, substract, substract_output, multiply, multiply_output

from keras.layers import Input, Lambda, merge
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = None


def build_model(embedding_matrix, max_length, hidden_unit, n_classes, keep_prob, load_pretrained_model=False):
    vocab_size, embedding_size = embedding_matrix.shape
    dropout_rate = 1 - keep_prob

    a = Input(shape=(max_length, ), dtype='int32', name='words_1')  # For "premise"
    b = Input(shape=(max_length, ), dtype='int32', name='words_2')  # For "hypothesis"

    # ------- Embedding Layer -------
    # Using "Glove" pre-trained embedding matrix as our initial weights
    embedding_layer = EmbeddingLayer(vocab_size, embedding_size, max_length, hidden_unit, init_weights=embedding_matrix, dropout=dropout_rate, nr_tune=5000)
    embedded_a = embedding_layer(a)
    embedded_b = embedding_layer(b)

    # ------- BiLSTM Layer -------
    # BiLSTM learns to represent a word and its context
    encoded_a = BiLSTM_Layer(max_length, hidden_unit)(embedded_a)
    encoded_b = BiLSTM_Layer(max_length, hidden_unit)(embedded_b)

    # ------- Attention Layer -------
    attention_ab = Lambda(attention, attention_output, name='attention')([encoded_a, encoded_b])

    # ------- Soft-Alignment Layer -------
    # Modeling local inference needs to employ some forms of hard or soft alignment to associate the relevant
    # sub-components between a premise and a hypothesis
    # Using inter-sentence "alignment" (or attention) to softly align each word to the content of hypothesis (or premise)
    align_alpha = Lambda(attention_softmax3d, attention_softmax3d_output, name='soft_alignment_a')([attention_ab, encoded_b])
    align_beta = Lambda(attention_softmax3d, attention_softmax3d_output, name='soft_alignment_b')([attention_ab, encoded_a])

    # ------- Enhancement Layer -------
    # Compute the difference and the element-wise product for the tuple < encoded_a, align_a > and < encoded_b, align_b >
    # This operation could help sharpen local inference information between elements in the tuples and capture
    # inference relationships such as contradiction.
    sub_a = Lambda(substract, substract_output, name='substract_a')([encoded_a, align_alpha])
    mul_a = Lambda(multiply, multiply_output, name='multiply_a')([encoded_a, align_alpha])

    sub_b = Lambda(substract, substract_output, name='substract_b')([encoded_b, align_beta])
    mul_b = Lambda(multiply, multiply_output, name='multiply_b')([encoded_b, align_beta])

    m_a = merge([encoded_a, align_alpha, sub_a, mul_a], mode='concat')  # shape=(batch_size, time-steps, 4 * units)
    m_b = merge([encoded_b, align_beta, sub_b, mul_b], mode='concat')  # shape=(batch_size, time-steps, 4 * units)

    # ------- Composition Layer -------
    comp_a = Composition_Layer(hidden_unit, max_length)(m_a)
    comp_b = Composition_Layer(hidden_unit, max_length)(m_b)

    # ------- Pooling Layer -------
    preds = Pooling_Layer(hidden_unit, n_classes, dropout=0.2, l2_weight_decay=1e-4)(comp_a, comp_b)

    model = Model(inputs=[a, b], outputs=[preds])
    model.compile(optimizer=Adam(lr=FLAGS.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    if load_pretrained_model:
        if FLAGS.load_model is None:
            raise ValueError("You need to specify the model location by --load_model=[location]")
        model.load_weights(FLAGS.load_model)

    return model


def do_predict(X=None):
    pass


def do_eval(test_data):
    if FLAGS.load_model is None:
        raise ValueError("You need to specify the model location by --load_model=[location]")

    # Load Testing Data
    question_1, question_2, labels = get_input_from_csv(test_data)

    # Load Pre-trained Model
    if FLAGS.best_glove:
        import en_core_web_md
        nlp = en_core_web_md.load()  # load best-matching version for Glove
    else:
        nlp = spacy.load('en')
    embedding_matrix = load_glove_embeddings(nlp.vocab, n_unknown=FLAGS.num_unknown)  # shape=(1071074, 300)
    model = build_model(embedding_matrix, FLAGS.max_length, FLAGS.num_hidden, FLAGS.num_classes, FLAGS.keep_prob, load_pretrained_model=True)

    # Convert the "raw data" to word-ids format && convert "labels" to one-hot vectors
    q1_test, q2_test = convert_questions_to_word_ids(question_1, question_2, nlp, max_length=FLAGS.max_length, tree_truncate=FLAGS.tree_truncate)
    labels = to_categorical(np.asarray(labels, dtype='int32'))

    accuracy = model.evaluate([q1_test, q2_test], labels, batch_size=FLAGS.batch_size, verbose=1)
    # print("[*] ACCURACY OF TEST DATA: %.4f" % accuracy)


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
    load_pretrained_model = True if FLAGS.load_model is not None else False
    model = build_model(embedding_matrix, FLAGS.max_length, FLAGS.num_hidden, FLAGS.num_classes, FLAGS.keep_prob, load_pretrained_model=load_pretrained_model)

    # Stage 4: Convert the "raw data" to word-ids format && convert "labels" to one-hot vectors
    tf.logging.info('Converting questions into ids ...')
    q1_train, q2_train = convert_questions_to_word_ids(train_question_1, train_question_2, nlp, max_length=FLAGS.max_length, tree_truncate=FLAGS.tree_truncate)
    train_labels = to_categorical(np.asarray(train_labels, dtype='int32'))

    # q1_val, q2_val = convert_questions_to_word_ids(val_question_1, val_question_2, nlp, max_length=FLAGS.max_length, tree_truncate=FLAGS.tree_truncate)
    # val_labels = to_categorical(np.asarray(val_labels, dtype='int32'))

    # Stage 5: Training
    tf.logging.info('Start training ...')

    save_dir = save_dir if save_dir is not None else 'checkpoints'
    filepath = os.path.join(save_dir, "weights-{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    model.fit(
        x=[q1_train, q2_train],
        y=train_labels,
        batch_size=batch_size,
        epochs=n_epochs,
        # validation_data=([q1_val, q2_val], val_labels),
        validation_split=0.2,
        callbacks=[checkpoint],
        shuffle=True,
        verbose=FLAGS.verbose
    )


def run(_):
    if FLAGS.mode == 'train':
        train(FLAGS.input_data, FLAGS.val_data, FLAGS.batch_size, FLAGS.num_epochs)
    elif FLAGS.mode == 'eval':
        do_eval(FLAGS.test_data)
    elif FLAGS.mode == 'predict':
        do_predict()
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
        '--mode',
        type=str,
        help='Specify mode: train or eval or predict',
        required=True
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)
