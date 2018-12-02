# -*- coding:utf-8 -*- 

from __future__ import division
from __future__ import print_function

from gensim.models import KeyedVectors

import numpy as np
import pandas as pd
import word2vec
import pickle
import os




# 往词向量中添加特殊字符集
SPECIAL_SYMBOL = ['<PAD>', '<EOS>']  


def get_word_embedding():
    """提取词向量，并保存至 ../data/word_embedding.npy"""
    print('getting the word_embedding.npy')
    file = 'E:/data/sgns.baidubaike.bigram-char'
    #file2 = 'E:/data/Tencent_AILab_ChineseEmbedding.txt'
    wv = KeyedVectors.load_word2vec_format(file, binary=False)
    word_embedding = wv.vectors # shape (411720, 256)
    words = wv.vocab    # shape (411720, ) # array(['</s>', 'w11', 'w54', ..., 'w133825'], dtype='<U78')
    
    n_special_sym = len(SPECIAL_SYMBOL) # 加入特殊词
    sr_id2word = pd.Series(words, index=range(n_special_sym, n_special_sym + len(words)))
    sr_word2id = pd.Series(range(n_special_sym, n_special_sym + len(words)), index=words)
    # 添加特殊符号：<PAD>:0, <UNK>:1 并随机初始化
    embedding_size = 256
    vec_special_sym = np.random.randn(n_special_sym, embedding_size)
    for i in range(n_special_sym):
        sr_id2word[i] = SPECIAL_SYMBOL[i]
        sr_word2id[SPECIAL_SYMBOL[i]] = i
    word_embedding = np.vstack([vec_special_sym, word_embedding]) # shape (411722, 256)
    
    # 保存词向量
    save_path = '../data/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + 'word_embedding.npy', word_embedding)
    
    # 保存词与id的对应关系
    #2    </s>
    #3     w11
    #4     w54
    #5      w6
    #6    w111    
    with open(save_path + 'sr_word2id.pkl', 'wb') as outp:
        pickle.dump(sr_id2word, outp)
        pickle.dump(sr_word2id, outp)
    print('Saving the word_embedding.npy to ../data/word_embedding.npy')


if __name__ == '__main__':
    get_word_embedding()
