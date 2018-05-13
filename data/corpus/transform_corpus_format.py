# encoding: utf-8
"""
@author: guuboi
@contact: guuboi@163.com
@time: 2018/4/27 下午10:18
"""
import os
import xml.etree.ElementTree as ET


def load_xml_corpus(file):
    """
    :param file: 语料样本集
    :param word2idx: 映射字典{word: index}
    :param max_seq_len: 限制语句序列的最大长度
    :return: texts-文本语句; targets-目标词组; labels-极性标签
    e.g.
    """

    if not os.path.isfile(file):
        raise Exception('courpus %s is not found' % file)

    tree = ET.parse(file)
    sentences = tree.getroot()
    texts = []
    targets = []
    labels = []

    for sentence in sentences:
        raw_text = sentence.find('text').text.strip().lower().split()
        for asp_terms in sentence.iter('aspectTerms'):
            for asp_term in asp_terms.findall('aspectTerm'):
                target_word = asp_term.get('term').lower()
                start = int(asp_term.get('from'))
                end = int(asp_term.get('to'))
                label = asp_term.get('polarity')
                label2idx = {'negative': -1, 'neutral': 0, 'positive': 1}
                label = label2idx.get(label)
                text = raw_text[:start] + ['$T$'] + raw_text[end:]
                texts.append(text)
                targets.append(target_word)
                labels.append(label)

    return texts, targets, labels


def transform_corpus_format(corpus, to_file):
    """
    :param corpus: load_xml_corpus返回的corpus
    :param to_file: 文件保存地址
    :return: None
    """
    text, target, label = corpus
    n = len(text)
    with open(to_file, 'w') as f:
        for i in range(n):
            f.write(' '.join(text[i]) + '\n')
            f.write(target[i] + '\n')
            f.write(str(label[i]) + '\n')


if __name__ == '__main__':
    file = './data/corpus/'
    train = load_xml_corpus(file + 'Auto_Train_v1.xml')
    test = load_xml_corpus(file + 'Auto_Test_v1.xml')
    transform_corpus_format(train, file+'Auto_Train_v1.txt')
    transform_corpus_format(test, file+'Auto_Test_v1.txt')

