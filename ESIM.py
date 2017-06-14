from layers import EmbeddingLayer, BiLSTM_Layer, Composition_Layer, Pooling_Layer, attention, attention_output, attention_softmax3d, attention_softmax3d_output, substract, substract_output, multiply, multiply_output

from keras.layers import Input, Lambda, merge
from keras.models import Model
from keras.optimizers import Adam


class ESIM:

    def __init__(self, embedding_matrix, max_length, hidden_unit, n_classes, keep_prob, learning_rate=1e-4, l2_weight_decay=1e-4):
        self.embedding_matrix = embedding_matrix
        self.vocab_size = embedding_matrix.shape[0]
        self.embedding_size = embedding_matrix.shape[1]
        self.dropout_rate = 1 - keep_prob
        self.max_length = max_length
        self.n_classes = n_classes
        self.hidden_unit = hidden_unit
        self.learning_rate = learning_rate

    def build_model(self, load_model=None):

        a = Input(shape=(self.max_length, ), dtype='int32', name='words_1')  # For "premise"
        b = Input(shape=(self.max_length, ), dtype='int32', name='words_2')  # For "hypothesis"

        # ------- Embedding Layer -------
        # Using "Glove" pre-trained embedding matrix as our initial weights
        embedding_layer = EmbeddingLayer(self.vocab_size, self.embedding_size, self.max_length, self.hidden_unit, init_weights=self.embedding_matrix, dropout=self.dropout_rate, nr_tune=5000)
        embedded_a = embedding_layer(a)
        embedded_b = embedding_layer(b)

        # ------- BiLSTM Layer -------
        # BiLSTM learns to represent a word and its context
        encoded_a = BiLSTM_Layer(self.max_length, self.hidden_unit)(embedded_a)
        encoded_b = BiLSTM_Layer(self.max_length, self.hidden_unit)(embedded_b)

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
        comp_a = Composition_Layer(self.hidden_unit, self.max_length)(m_a)
        comp_b = Composition_Layer(self.hidden_unit, self.max_length)(m_b)

        # ------- Pooling Layer -------
        preds = Pooling_Layer(self.hidden_unit, self.n_classes, dropout=self.dropout_rate, l2_weight_decay=self.l2_weight_decay)(comp_a, comp_b)

        model = Model(inputs=[a, b], outputs=[preds])
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

        if load_model is not None:
            print('Loading pre-trained weights from \'{}\'...'.format(load_model))
            model.load_weights(load_model)

        return model
