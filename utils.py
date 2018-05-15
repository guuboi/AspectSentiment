# encoding: utf-8
"""
@author: guuboi
@contact: guuboi@163.com
@time: 2018/4/27 下午10:18
"""
import numpy as np
import time
from datetime import timedelta
import tensorflow.contrib.keras as kr


def time_diff(start_time):
    """当前距初始时间已花费的时间"""
    end_time = time.time()
    diff = end_time - start_time
    return timedelta(seconds=int(round(diff)))


def batch_index(length, batch_size, is_shuffle=True):
    """
    生成批处理样本序列id.
    :param length: 样本总数
    :param batch_size: 批处理大小
    :param is_shuffle: 是否打乱样本顺序
    :return:
    """
    index = [idx for idx in range(length)]
    if is_shuffle:
        np.random.shuffle(index)
    for i in range(int(np.ceil(length / batch_size))):
        yield index[i * batch_size:(i + 1) * batch_size]


def cat_to_id(classes=None):
    """
    :param classes: 分类标签；默认为pos, neg
    :return: {分类标签：id}
    """
    if not classes:
        classes = ['1', '-1']
    cat2id = {cat: idx for (idx, cat) in enumerate(classes)}
    return classes, cat2id


def load_corpus(file, word2id, max_sen_len, max_target_len=2):
    cat, cat2id = cat_to_id()
    left_context, left_length = [], []
    right_context, right_length = [], []
    labels = []
    target_words = []
    lines = open(file, encoding='utf-8').readlines()

    for i in range(0, len(lines), 3):
        target_word = lines[i + 1].strip().lower().split()
        target_word = [word2id.get(w, 0) for w in target_word]
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
                words_l.append(word2id.get(word, 0))
            else:
                words_r.append(word2id.get(word, 0))

        words_l = words_l[1-max_sen_len:]
        words_l.extend(target_word)
        #######修改#############
        left_length.append(len(words_l))
        left_context.append(words_l + [0] * (max_sen_len - len(words_l)))

        words_r = words_r[:max_sen_len-1]
        tmp = target_word + words_r
        tmp.reverse()
        right_length.append(len(tmp))
        right_context.append(tmp + [0] * (max_sen_len - len(tmp)))

    # onehot编码
    labels = [cat2id[l] for l in labels]
    labels = kr.utils.to_categorical(labels, len(cat2id))

    # target_words处理为定长：[[1, 2], [1], [2, 3]] => [[1, 2], [1, 1], [2, 3]]
    target_words = [w + (max_target_len - len(w)) * w for w in target_words]
    return np.asarray(left_context), np.asarray(left_length), np.asarray(right_context), \
           np.asarray(right_length), np.asarray(target_words), np.asarray(labels)


def build_word2id(files, to_file):
    """
    :param files: 语料库文件列表
    :param to_file: word2id保存地址
    :return: None
    """
    word2id = {'_PAD_': 0}
    for _path in files:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                for word in line.strip().split():
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)

    with open(to_file, 'w', encoding='utf-8') as f:
        for w in word2id:
            f.write(w+'\t')
            f.write(str(word2id[w]))
            f.write('\n')


# files = ['./data/auto_aspect_train.txt', './data/auto_aspect_test.txt']
# build_word2id(files, './data/word_to_id.txt')


def load_word2id(path):
    """
    :param path: word_to_id词汇表路径
    :return: word_to_id:{word: id}
    """
    word_to_id = {}
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            word = sp[0]
            idx = int(sp[1])
            if word not in word_to_id:
                word_to_id[word] = idx
    return word_to_id


def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    import gensim
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs

# word2id = load_word2id('./data/word_to_id.txt')
# w2v = build_word2vec('./data/wiki_word2vec_50.bin', word2id, save_to_path='./data/corpus_word2vec.txt')


def load_corpus_word2vec(path):
    """加载语料库word2vec词向量,相对wiki词向量相对较小"""
    word2vec = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = [float(w) for w in line.strip().split()]
            word2vec.append(sp)
    return np.asarray(word2vec)


# word2id = load_word2id('./data/word_to_id.txt')
# w2v = load_corpus_word2vec('./data/corpus_word2vec.txt')
# train = load_corpus('./data/auto_aspect_train.txt', word2id, 30, max_target_len=2)

