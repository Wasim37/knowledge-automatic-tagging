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
    print('geting the word embedding...')
    file = 'E:/data/sgns.baidubaike.bigram-char'
    wv = KeyedVectors.load_word2vec_format(file, binary=False)
    word_embedding = wv.vectors # shape (635974, 300)
    words = wv.vocab.keys()    # len (635974, )
    
    # 添加特殊符号：<PAD>:0, <UNK>:1 并随机初始化
    n_special_sym = len(SPECIAL_SYMBOL) # 加入特殊词
    embedding_size = 300
    vec_special_sym = np.random.randn(n_special_sym, embedding_size)
    word_embedding = np.vstack([vec_special_sym, word_embedding]) # shape (635976, 300)
    
    # 保存词向量
    save_path = '../data/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + 'word_embedding.npy', word_embedding)
    print('Saving the word_embedding.npy to ../data/word_embedding.npy')
    
    # 保存词与id的对应关系
    sr_id2word = pd.Series(list(words), index=range(n_special_sym, n_special_sym + len(words)))
    sr_word2id = pd.Series(range(n_special_sym, n_special_sym + len(words)), index=list(words))
    for i in range(n_special_sym):
        sr_id2word[i] = SPECIAL_SYMBOL[i]
        sr_word2id[SPECIAL_SYMBOL[i]] = i   
        
    with open(save_path + 'sr_word2id.pkl', 'wb') as outp:
        pickle.dump(sr_id2word, outp)
        pickle.dump(sr_word2id, outp)
        

if __name__ == '__main__':
    get_word_embedding()
