# encoding: utf-8
import tensorflow as tf
from utils import batch_index, load_word2vec, load_coupus


class Config:
    def __init__(self):
        self.n_class = 3
        self.max_target_len = 2
        self.max_sentence_len = 40
        self.embedding_dim = 50
        self.batch_size = 100
        self.n_hidden = 100
        self.n_epoch = 10
        self.learning_rate = 0.01
        self.drop_keep_prob = 0.6
        self.l2_reg = 0.001
        self.train_file_path = './data/corpus/Auto_Train_v1.txt'
        self.test_file_path = './data/corpus/Auto_Test_v1.txt'
        self.embedding_file_path = './data/wiki_word2vec_50.bin'


class LSTM(object):
    def __init__(self, config, embeddings):
        self.max_target_len = config.max_target_len
        self.embeddings = tf.constant(embeddings, dtype=tf.float32)
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.n_hidden = config.n_hidden
        self.learning_rate = config.learning_rate
        self.n_class = config.n_class
        self.max_sentence_len = config.max_sentence_len
        self.l2_reg = config.l2_reg
        self.n_epoch = config.n_epoch
        self.dropout_keep_prob = config.drop_keep_prob

        self.x = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='left_corpus')
        self.sen_len = tf.placeholder(tf.int32, None, name='left_corpus_length')

        self.x_bw = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='right_corpus')
        self.sen_len_bw = tf.placeholder(tf.int32, [None], name='right_corpus_length')

        self.target_words = tf.placeholder(tf.int32, [None, self.max_target_len], name='target_words')
        self.y = tf.placeholder(tf.int32, [None, self.n_class], name='global_label')

        with tf.name_scope('weights'):
            self.weights = {
                'softmax_bi_lstm': tf.get_variable(
                    name='bi_lstm_w',
                    shape=[2 * self.n_hidden, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.003, 0.003),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax_bi_lstm': tf.get_variable(
                    name='bi_lstm_b',
                    shape=[self.n_class],
                    initializer=tf.random_uniform_initializer(-0.003, 0.003),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
            }

        self.build()

    def bi_dynamic_lstm(self):
        """
        :params: self.x, self.x_bw, self.seq_len, self.seq_len_bw,
                self.weights['softmax_lstm'], self.biases['softmax_lstm']
        :return: non-norm prediction values
        """
        inputs_fw, inputs_bw = self.add_embeddings()
        inputs_fw = tf.nn.dropout(inputs_fw, keep_prob=self.dropout_keep_prob)
        inputs_bw = tf.nn.dropout(inputs_bw, keep_prob=self.dropout_keep_prob)

        with tf.name_scope('forward_lstm'):
            outputs_fw, state_fw = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.n_hidden),
                inputs=inputs_fw,
                sequence_length=self.sen_len,
                dtype=tf.float32,
                scope='LSTM_fw'
            )
            batch_size = tf.shape(outputs_fw)[0]
            index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len - 1)
            output_fw = tf.gather(tf.reshape(outputs_fw, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        with tf.name_scope('backward_lstm'):
            outputs_bw, state_bw = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.n_hidden),
                inputs=inputs_bw,
                sequence_length=self.sen_len_bw,
                dtype=tf.float32,
                scope='LSTM_bw'
            )
            batch_size = tf.shape(outputs_bw)[0]
            index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len_bw - 1)
            output_bw = tf.gather(tf.reshape(outputs_bw, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        output = tf.concat([output_fw, output_bw], 1)  # batch_size * 2n_hidden
        predict = tf.matmul(output, self.weights['softmax_bi_lstm']) + self.biases['softmax_bi_lstm']
        return predict

    def add_embeddings(self):
        """
        :return:
        """
        fw_words = tf.nn.embedding_lookup(self.embeddings, self.x)
        bw_words = tf.nn.embedding_lookup(self.embeddings, self.x_bw)
        batch_size = tf.shape(bw_words)[0]
        target_words = tf.reduce_mean(tf.nn.embedding_lookup(self.embeddings, self.target_words), 1, keep_dims=True)
        target_words = tf.zeros([batch_size, self.max_sentence_len, self.embedding_dim], tf.float32) + target_words
        fw_words = tf.concat([fw_words, target_words], 2)
        bw_words = tf.concat([bw_words, target_words], 2)
        return fw_words, bw_words

    def loss(self, pred):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y)
        loss = tf.reduce_mean(loss)
        return loss

    def optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optimizer

    def correct_num(self, pred):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        correct_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
        # _acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return correct_num

    def build(self):
        self.pred = self.bi_dynamic_lstm()
        self.loss = self.loss(self.pred)
        self.optimizer = self.optimizer(self.loss)
        self.correct_num = self.correct_num(self.pred)

    def get_batch_data(self, x, sen_len, x_bw, sen_len_bw, target_words, y=None, batch_size=100):
        for index in batch_index(len(x), batch_size, 1):
            feed_dict = {
                self.x: x[index],
                self.x_bw: x_bw[index],
                self.sen_len: sen_len[index],
                self.sen_len_bw: sen_len_bw[index],
                self.target_words: target_words[index],
            }
            if y is not None:
                feed_dict[self.y] = y[index]
            yield feed_dict

    def train_on_batch(self, sess, feed):
        _, loss, n_correct = sess.run([self.optimizer, self.loss, self.correct_num], feed_dict=feed)
        return loss, n_correct

    def test_on_batch(self, sess, feed):
        loss, n_correct = sess.run([self.loss, self.correct_num], feed_dict=feed)
        return loss, n_correct

    def predict_on_batch(self, sess, feed):
        pred = sess.run(self.pred, feed_dict=feed)
        prob = tf.nn.softmax(logits=pred, dim=1)
        return prob

    def fit(self, sess, train_set, test_set):
        tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word = train_set
        te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word = test_set
        n_train = len(tr_x)
        n_test = len(te_x)
        max_acc = 0.

        for epoch in range(self.n_epoch):
            print('Epoch {} ########'.format(epoch + 1))
            tr_acc, tr_loss = 0., 0.
            for _train in self.get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw,
                                             tr_target_word, tr_y, self.batch_size):
                _tr_loss, _tr_acc = self.train_on_batch(sess, _train)
                tr_loss += _tr_loss
                tr_acc += _tr_acc

            tr_loss /= n_train
            tr_acc /= n_train

            te_acc, te_loss = 0., 0.
            for _test in self.get_batch_data(te_x, te_sen_len, te_x_bw, te_sen_len_bw,
                                            te_target_word, te_y, self.batch_size):

                _te_loss, _te_acc = self.test_on_batch(sess, _test)
                te_loss += _te_loss
                te_acc += _te_acc

            te_loss /= n_test
            te_acc /= n_test

            print('Train: loss={}, accuracy={}\nTest: loss={}, accuracy={}'.format(tr_loss, tr_acc, te_loss, te_acc))

        print('Optimization Finished! Max acc={}'.format(max_acc))
        print('Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
            self.learning_rate,
            self.n_epoch,
            self.batch_size,
            self.n_hidden,
            self.l2_reg
        ))


config = Config()
word2id = {}
train = load_coupus(config.train_file_path, word2id, config.max_sentence_len)
test = load_coupus(config.test_file_path, word2id, config.max_sentence_len)
w2c = load_word2vec(config.embedding_file_path, word2id)
lstm = LSTM(config, w2c)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    lstm.fit(sess, train, test)



