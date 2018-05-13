# encoding: utf-8
"""
@author: guuboi
@contact: guuboi@163.com
@time: 2018/4/27 下午10:18
"""
import numpy as np
import gensim


def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    """
    生成批处理样本序列id.
    :param length: 样本总数
    :param batch_size: 批处理大小
    :param n_iter: 训练迭代次数
    :param is_shuffle: 是否打乱样本顺序
    :return:
    """
    index = [idx for idx in range(length)]
    for j in range(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in range(int(np.ceil(length / batch_size))):
            yield index[i * batch_size:(i + 1) * batch_size]


def onehot_encoder(label):
    """
    one编码{1: {1, 0, 0}, 0: {0, 1,0 }, -1: {0, 0, 1}
    :param label: 极性标签(-1/0/1)
    :return: 3位onehot编码
    """
    if label == '1':
        return [1, 0, 0]
    elif label == '0':
        return [0, 1, 0]
    else:
        return [0, 0, 1]


def load_coupus(file, word2id, sentence_len, max_target_len=2):
    left_context, left_length = [], []
    right_context, right_length = [], []
    labels = []
    target_words = []
    lines = open(file).readlines()

    for i in range(0, len(lines), 3):
        target_word = lines[i + 1].strip().lower().split()

        for w in target_word:
            if w not in word2id.keys():
                word2id[w] = len(word2id)
        target_word = [word2id[w] for w in target_word]
        target_words.append(target_word[:max_target_len])

        labels.append(lines[i + 2].strip())

        words = lines[i].strip().lower().split()
        words_l, words_r = [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word not in word2id:
                    word2id[word] = len(word2id)

                words_l.append(word2id[word])
            else:
                if word not in word2id:
                    word2id[word] = len(word2id)

                words_r.append(word2id[word])

        words_l.extend(target_word)
        left_length.append(len(words_l))
        left_context.append(words_l + [0] * (sentence_len - len(words_l)))
        tmp = target_word + words_r
        tmp.reverse()
        right_length.append(len(tmp))
        right_context.append(tmp + [0] * (sentence_len - len(tmp)))

    # onehot编码
    labels = [onehot_encoder(l) for l in labels]
    # target_words处理为定长：[[1, 2], [1], [2, 3]] => [[1, 2], [1, 1], [2, 3]]
    target_words = [w + (max_target_len - len(w)) * w for w in target_words]
    return np.asarray(left_context), np.asarray(left_length), np.asarray(right_context), \
           np.asarray(right_length), np.asarray(labels), np.asarray(target_words)


def load_word2vec(fname, word2id):
    """
    加载预训练的word2vec模型.
    :param fname: 预训练的word2vec.
    :param vocab: 语料文本中包含的词汇集.
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    # n_words = len(word2id)
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    return word_vecs
