# encoding: utf-8

class CONFIG:
    """模型训练or预测的一些参数配置"""
    vocab_size = 4217
    n_class = 2            # 分类类别
    max_target_len = 2     # target词的最大组成词数：例如： 内饰：1；后排 空间：2
    max_sen_len = 40       # 上下文词最大数：左侧或右侧
    embedding_dim = 50     # 词向量维数
    batch_size = 100       # 批处理大小
    n_hidden = 200         # 隐藏层节点数
    n_epoch = 20           # 训练迭代数
    opt = 'adam'           # 训练优化器：adam或者adadelta
    learning_rate = 0.001  # 学习率
    l2_reg = 0.001         # l2正则
    drop_keep_prob = 0.6   # dropout
    update_w2v = True      # 是否在训练中微调word2vec
    print_per_batch = 1
    save_dir = './checkpoints/'
    word2id_file = './data/word_to_id.txt'
    train_file = './data/auto_aspect_train.txt'
    test_file = './data/auto_aspect_test.txt'
    pre_w2v_file = './data/wiki_word2vec_50.bin'
    corpus_w2v_file = './data/corpus_word2vec.txt'


