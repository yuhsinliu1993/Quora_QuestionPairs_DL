import numpy as np
import re
import spacy


def get_cleaned_text(text):
    """
    Stage of preprocessing the dirty texts
    """
    return text


def load_glove_embeddings(vocab, n_unknown=100):
    if not isinstance(vocab, spacy.vocab.Vocab):
        raise TypeError("The input `vocab` must be type of 'spacy.vocab.Vocab', not %s." % type(vocab))

    max_vector_length = max(lex.rank for lex in vocab) + 1  # index start from 1
    matrix = np.zeros((max_vector_length + n_unknown + 2, vocab.vectors_length), dtype='float32')  # 2 for <PAD> and <EOS>

    # Normalization
    for lex in vocab:
        if lex.has_vector:
            matrix[lex.rank + 1] = lex.vector / lex.vector_norm

    return matrix


def _get_word_ids(docs, rnn_encode=True, tree_truncate=False, max_length=100, nr_unk=100):
    Xs = np.zeros((len(docs), max_length), dtype='int32')

    for i, doc in enumerate(docs):
        if tree_truncate:
            if isinstance(doc, Span):
                queue = [doc.root]
            else:
                queue = [sent.root for sent in doc.sents]
        else:
            queue = list(doc)
        words = []
        while len(words) <= max_length and queue:
            word = queue.pop(0)
            if rnn_encode or (not word.is_punct and not word.is_space):
                words.append(word)
            if tree_truncate:
                queue.extend(list(word.lefts))
                queue.extend(list(word.rights))
        words.sort()
        for j, token in enumerate(words):
            if token.has_vector:
                Xs[i, j] = token.rank + 1
            else:
                Xs[i, j] = (token.shape % (nr_unk - 1)) + 2
            j += 1
            if j >= max_length:
                break
        else:
            Xs[i, len(words)] = 1
    return Xs


def convert_questions_to_word_ids(question_1, question_2, nlp, max_length, n_threads=10, batch_size=128, tree_truncate=False):
    Xs = []
    for texts in (question_1, question_2):
        Xs.append(_get_word_ids(list(nlp.pipe(texts, n_threads=n_threads, batch_size=batch_size)),
                                max_length=max_length,
                                tree_truncate=tree_truncate))

    return Xs[0], Xs[1]


def to_categorical(y, nb_classes=None):
    y = np.asarray(y, dtype='int32')

    if not nb_classes:
        nb_classes = np.max(y) + 1

    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.

    return Y
