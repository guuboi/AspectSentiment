# encoding: utf-8
"""
@author: guuboi
@contact: guuboi@163.com
@time: 2018/4/27 下午10:18
"""
import os
import time
import numpy as np
import tensorflow as tf
from utils import time_diff, batch_index


class LSTM(object):
    def __init__(self, config, embeddings=None):
        self.vocab_size = config.vocab_size
        self.update_w2v = config.update_w2v
        self.max_target_len = config.max_target_len
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.n_hidden = config.n_hidden
        self.learning_rate = config.learning_rate
        self.n_class = config.n_class
        self.max_sen_len = config.max_sen_len

        self.opt = config.opt
        self.l2_reg = config.l2_reg
        self.n_epoch = config.n_epoch
        self.dropout_keep_prob = config.drop_keep_prob

        # self.embeddings = tf.constant(embeddings, dtype=tf.float32)
        if embeddings is not None:
            self.word_embeddings = tf.Variable(embeddings, dtype=tf.float32, trainable=self.update_w2v)
        else:
            self.word_embeddings = tf.Variable(
                tf.zeros([self.vocab_size, self.embedding_dim]),
                dtype=tf.float32,
                trainable=self.update_w2v)

        # x: forword context的各个词的id号
        # sen_len: forword context的实际词数
        self.x = tf.placeholder(tf.int32, [None, self.max_sen_len], name='left_corpus')
        self.sen_len = tf.placeholder(tf.int32, None, name='left_corpus_length')

        # x_bw: backword context的各个词的id号
        # sen_len_bw: backword context的实际次数
        self.x_bw = tf.placeholder(tf.int32, [None, self.max_sen_len], name='right_corpus')
        self.sen_len_bw = tf.placeholder(tf.int32, [None], name='right_corpus_length')

        # target_words: target词的id号
        # y: 输出类别
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
        :return: 未经过softmax转换的输出。
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
            index = tf.range(0, batch_size) * self.max_sen_len + (self.sen_len - 1)
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
            index = tf.range(0, batch_size) * self.max_sen_len + (self.sen_len_bw - 1)
            output_bw = tf.gather(tf.reshape(outputs_bw, [-1, self.n_hidden]), index)  # batch_size * n_hidden

        output = tf.concat([output_fw, output_bw], 1)  # batch_size * 2n_hidden
        predict = tf.matmul(output, self.weights['softmax_bi_lstm']) + self.biases['softmax_bi_lstm']
        return predict

    def add_embeddings(self):
        """输入x以及x_bw，转换为词向量"""
        fw_words = tf.nn.embedding_lookup(self.word_embeddings, self.x)
        bw_words = tf.nn.embedding_lookup(self.word_embeddings, self.x_bw)
        batch_size = tf.shape(bw_words)[0]
        target_words = tf.reduce_mean(tf.nn.embedding_lookup(self.word_embeddings, self.target_words), 1, keep_dims=True)
        target_words = tf.zeros([batch_size, self.max_sen_len, self.embedding_dim], tf.float32) + target_words
        fw_words = tf.concat([fw_words, target_words], 2)
        bw_words = tf.concat([bw_words, target_words], 2)
        return fw_words, bw_words

    def add_loss(self, pred):
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y)
        cost = tf.reduce_mean(cost)
        return cost

    def add_optimizer(self, loss):
        if self.opt == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-6)
        else:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        opt = optimizer.minimize(loss)
        return opt

    def add_accuracy(self, pred):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy

    def get_batches(self, X, y=None, batch_size=100, shuffle=True):
        x, sen_len, x_bw, sen_len_bw, target_words = X
        for index in batch_index(len(x), batch_size, shuffle):
            n = len(index)
            feed_dict = {
                self.x: x[index],
                self.x_bw: x_bw[index],
                self.sen_len: sen_len[index],
                self.sen_len_bw: sen_len_bw[index],
                self.target_words: target_words[index],
            }
            if y is not None:
                feed_dict[self.y] = y[index]
            yield feed_dict, n

    def build(self):
        self.pred = self.bi_dynamic_lstm()
        self.loss = self.add_loss(self.pred)
        self.optimizer = self.add_optimizer(self.loss)
        self.accuracy = self.add_accuracy(self.pred)

    def train_on_batch(self, sess, feed):
        _, _loss, _acc = sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=feed)
        return _loss, _acc

    def test_on_batch(self, sess, feed):
        _loss, _acc = sess.run([self.loss, self.accuracy], feed_dict=feed)
        return _loss, _acc

    def predict_on_batch(self, sess, feed, prob=True):
        result = tf.argmax(self.pred, 1)
        if prob:
            result = tf.nn.softmax(logits=self.pred, dim=1)

        res = sess.run(result, feed_dict=feed)
        return res

    def predict(self, sess, X, prob=False):
        yhat = []
        for _feed, _ in self.get_batches(X, batch_size=self.batch_size, shuffle=False):
            _yhat = self.predict_on_batch(sess, _feed, prob)
            yhat += _yhat.tolist()
        return np.array(yhat)

    def evaluate(self, sess, X, y):
        """评估在某一数据集上的准确率和损失"""
        num = len(y)
        total_loss, total_acc = 0., 0.
        for _feed, _n in self.get_batches(X, y, batch_size=self.batch_size):
            loss, acc = self.test_on_batch(sess, _feed)
            total_loss += loss * _n
            total_acc += acc * _n
        return total_loss / num, total_acc / num

    def fit(self, sess, X_train, y_train, X_dev, y_dev, save_dir=None, print_per_batch=100):
        saver = tf.train.Saver()
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        sess.run(tf.global_variables_initializer())

        print('Training and evaluating...')
        start_time = time.time()
        total_batch = 0  # 总批次
        best_acc_dev = 0.0  # 最佳验证集准确率
        last_improved = 0  # 记录上次提升批次
        require_improvement = 500  # 如果超过500轮模型效果未提升，提前结束训练
        flags = False
        for epoch in range(self.n_epoch):
            print('Epoch:', epoch + 1)
            for train_feed, train_n in self.get_batches(X_train, y_train, batch_size=self.batch_size):
                loss_train, acc_train = self.train_on_batch(sess, train_feed)
                loss_dev, acc_dev = self.evaluate(sess, X_dev, y_dev)

                if total_batch % print_per_batch == 0:
                    if acc_dev > best_acc_dev:
                        # 保存在验证集上性能最好的模型
                        best_acc_dev = acc_dev
                        last_improved = total_batch
                        if save_dir:
                            saver.save(sess=sess, save_path=os.path.join(save_dir, 'sa-model'))
                        improved_str = '*'
                    else:
                        improved_str = ''

                    time_dif = time_diff(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' + \
                          ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_dev, acc_dev, time_dif, improved_str))
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    print('No optimization for a long time, auto-stopping...')
                    flags = True
                    break
            if flags:
                break




