"""
    Based on the paper "A decomposable attention model for natural language inference"
    See more detail on https://arxiv.org/abs/1606.01933
"""

import keras.backend as K
from keras.layers import Dense, merge
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed, SpatialDropout1D
from keras.layers import Bidirectional, GRU, LSTM
from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import ELU
from keras.models import Sequential, Model, model_from_json
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D


class EmbeddingLayer(object):

    def __init__(self, vocab_size, embedding_size, max_length, output_units, init_weights=None, nr_tune=1000, dropout=0.0):
        self.output_units = output_units
        self.max_length = max_length

        self.embed = Embedding(
            vocab_size,
            embedding_size,
            input_length=max_length,
            weights=[init_weights],
            name='embedding',
            trainable=False)

        self.tune = Embedding(
            nr_tune,
            output_units,
            input_length=max_length,
            weights=None,
            name='tune',
            trainable=True,
            dropout=dropout)

        self.mod_ids = Lambda(lambda sent: sent % (nr_tune - 1) + 1,
                              output_shape=(self.max_length,))

        self.project = TimeDistributed(Dense(output_units, activation=None, bias=False, name='project'))

    def __call__(self, sentence):

        def get_output_shape(shapes):
            print(shapes)

            return shapes[0]

        mod_sent = self.mod_ids(sentence)
        tuning = self.tune(mod_sent)
        # tuning = merge([tuning, mod_sent],
        #    mode=lambda AB: AB[0] * (K.clip(K.cast(AB[1], 'float32'), 0, 1)),
        #    output_shape=(self.max_length, self.output_units))
        pretrained = self.project(self.embed(sentence))
        vectors = merge([pretrained, tuning], mode='sum')
        return vectors


class BiRNN_EncodingLayer(object):

    def __init__(self, max_length, hidden_units, dropout=0.0):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(hidden_units, return_sequences=True, dropout_W=dropout, dropout_U=dropout), input_shape=(max_length, hidden_units)))
        self.model.add(TimeDistributed(Dense(hidden_units, activation='relu', init='he_normal')))
        self.model.add(TimeDistributed(Dropout(0.2)))

    def __call__(self, embedded_words):
        return self.model(embedded_words)


class AttentionLayer(object):

    def __init__(self, max_length, hidden_units, dropout=0.0, L2=0.0, activation='relu'):
        self.max_length = max_length
        self.model = Sequential()

        """
        F function => attention = transpose of F(a) * F(b)
        """
        self.model.add(Dropout(dropout, input_shape=(hidden_units,)))
        self.model.add(Dense(hidden_units,
                             name='attend1',
                             init='he_normal',
                             W_regularizer=l2(L2),
                             input_shape=(hidden_units,),
                             activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(hidden_units, name='attend2',
                             init='he_normal', W_regularizer=l2(L2), activation='relu'))
        self.model = TimeDistributed(self.model)

    def __call__(self, sent1, sent2):
        def _outer(AB):
            """
            Calculate unnormalized attention weights
            """
            energy = K.batch_dot(x=AB[1], y=K.permute_dimensions(AB[0], pattern=(0, 2, 1)))
            return K.permute_dimensions(energy, (0, 2, 1))

        return merge(inputs=[self.model(sent1), self.model(sent2)],
                     mode=_outer,
                     output_shape=(self.max_length, self.max_length))


class SoftAlignmentLayer(object):

    def __init__(self, max_length, hidden_units):
        self.max_length = max_length
        self.hidden_units = hidden_units

    def __call__(self, sentence, attention, transpose=False):

        def _normalize_attention(attention_and_sent):
            attention = attention_and_sent[0]  # attention matrix   shape=(?, max_length, max_length)
            sentence = attention_and_sent[1]  # sentence that wants to be aligned   shape=(?, max_length, embedding_size)

            if transpose:
                attention = K.permute_dimensions(attention, (0, 2, 1))

            # 3D softmax - calculate the subphrase in the sentence through attention
            exp = K.exp(attention - K.max(attention, axis=-1, keepdims=True))
            summation = K.sum(exp, axis=-1, keepdims=True)
            weights = exp / summation  # (512, 512)
            subphrase_in_sentence = K.batch_dot(weights, sentence)

            return subphrase_in_sentence

        return merge([attention, sentence],
                     mode=_normalize_attention,
                     output_shape=(self.max_length, self.hidden_units))


class ComparisonLayer(object):
    """
    Separately compare the aligned phrases using a function "G"
    """

    def __init__(self, words, hidden_units, L2=0.0, dropout=0.0):
        self.words = words

        self.model = Sequential()
        self.model.add(Dropout(dropout, input_shape=(hidden_units * 2,)))  # 2: for sentence and
        self.model.add(Dense(hidden_units, name='compare1', init='he_normal', W_regularizer=l2(L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(hidden_units, name='compare2', init='he_normal', W_regularizer=l2(L2)))
        self.model.add(Activation('relu'))
        self.model = TimeDistributed(self.model)

    def __call__(self, sent, align, **kwargs):
        result = self.model(merge([sent, align], mode='concat'))  # Shape: (batch, max_length, 2 * hidden_units)

        # avged = GlobalAveragePooling1D()(result, mask=self.words)
        avged = GlobalAveragePooling1D()(result)
        # maxed = GlobalMaxPooling1D()(result, mask=self.words)
        maxed = GlobalMaxPooling1D()(result)
        merged = merge([avged, maxed], mode='sum')
        result = BatchNormalization()(merged)

        return result


class AggregationLayer(object):
    """
    Concatenate two sets of comparison vectors and aggregate over each set by summation
    y = H([v1, v2])
    """

    def __init__(self, hidden_units, output_units, dropout=0.0, L2=0.0):
        # Define H function
        self.model = Sequential()
        self.model.add(Dropout(dropout, input_shape=(hidden_units * 2,)))
        self.model.add(Dense(hidden_units, name='entail_1', init='he_normal', W_regularizer=l2(L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(hidden_units, name='entail_2', init='he_normal', W_regularizer=l2(L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(output_units, name='entail_out', activation='softmax', W_regularizer=l2(L2), init='zero'))

    def __call__(self, feats1, feats2):
        predictions = self.model(merge([feats1, feats2], mode='concat'))
        return predictions
