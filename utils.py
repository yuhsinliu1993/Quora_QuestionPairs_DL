import numpy as np
import re
import spacy

stopwords = ['the', 'just', 'being', 'through', 'its', 'll', 'had', 'to', 'has', 'this', 'he', 'she', 'each', 'further', 'few', 'doing', 'for', 're', 'be', 'we', 'hers', 'by', 'on', 'about', 'of', 'or', 'own', 'into', 'mightn', 'your', 'from', 'her', 'their', 'there', 'been', 'whom', 'too', 'that', 'with', 'those', 'ma', 'will', 'theirs', 'and', 've', 'then', 'am', 'it', 'an', 'as', 'itself', 'at', 'in', 'any', 'if', 'other', 'you', 'shan', 'such', 'a', 'off', 'i', 'm', 'so', 'y', 'having']


def get_cleaned_text(text, remove_stop_words=True):

    if remove_stop_words:
        text = text.lower().split()
        text = [w for w in text if w not in stopwords]
        text = " ".join(text)

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=_\']", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\bm\b", " ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\be g\b", " eg ", text)
    text = re.sub(r"\bb g\b", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"\b9 11\b", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\bidk\b", "i do not know ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r"\bomg\b", " oh my god ", text)
    text = re.sub(r"\busa\b", " america ", text)
    text = re.sub(r"\bUSA\b", " america ", text)
    text = re.sub(r"\bu s\b", " america ", text)
    text = re.sub(r"\buk\b", " england ", text)
    text = re.sub(r"\bUK\b", " england ", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"\bdms\b", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r"\bcs\b", " computer science ", text)
    text = re.sub(r"\bupvotes\b", " up votes ", text)
    text = re.sub(r"\biPhone\b", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"the us", "america", text)
    text = re.sub(r"\bj k\b", " jk ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ", text)
    text = re.sub(r"\+", " ", text)
    text = re.sub(r"\-", " ", text)
    text = re.sub(r"\=", " ", text)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r"\s{2,}", " ", text)

    if remove_stop_words:
        text = text.split()
        text = [w for w in text if w not in stopwords]
        text = " ".join(text)

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


def shuffle_data(question_1, question_2, labels):
    q1 = []
    q2 = []
    y = []

    shuffle_indices = np.random.permutation(np.arange(len(question_1)))
    for index in shuffle_indices:
        q1.append(question_1[index])
        q2.append(question_2[index])
        y.append(labels[index])

    return q1, q2, y
