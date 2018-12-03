# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pickle
import jieba
from multiprocessing import Pool
from tqdm import tqdm
import time

save_path = '../data/'
with open(save_path + 'sr_word2id.pkl', 'rb') as inp:
    sr_id2word = pickle.load(inp)
    sr_word2id = pickle.load(inp)
dict_word2id = dict()
for i in range(len(sr_word2id)):
    dict_word2id[sr_word2id.index[i]] = sr_word2id.values[i]


def get_id(word):
    """获取 word 所对应的 id.
    如果该词不在词典中，用 <UNK>（对应的 ID 为 1 ）进行替换。
    """
    if word not in dict_word2id:
        return 1
    else:
        return dict_word2id[word]


def get_id4words(words):
    """把 words 转为 对应的 id"""
    cut = jieba.cut(words.strip()) # jieba分词
    words = ','.join(cut).split(',') 
    # python3种 map()和filter()返回的不是list，如果要转换为列表，可以使用list()转换
    # https://blog.csdn.net/qq_42397914/article/details/81586060    
    ids = list(map(get_id, words))  # 获取id
    return ids


def train_word2id():
    """把训练集的所有词转成对应的id。"""
    time0 = time.time()
    print('Processing train data.')
    df_trains = pd.read_csv('../raw_data/question_set.txt', sep='\t', usecols=[0, 1],
                            iterator = True,  # 当数据量很大时，可以返回一个迭代对象 TextFileReader，通过 get_chunk 取特定行的数据 
                            names=['question_id', 'word_content'], dtype={'question_id': object})
    
    # 数据太大，只读取前100W行（约三分之一数据）
    df_train = df_trains.get_chunk(1000000)
    print('training question number %d ' % len(df_train))  
    
    # 转为 id 形式
    p = Pool()
    train_content = np.asarray(list(p.map(get_id4words, df_train.word_content.values)))
    np.save('../data/wd_train_content.npy', train_content)
    p.close()
    p.join()
    print('Finished changing the training words to ids. Costed time %g s' % (time.time() - time0))


if __name__ == '__main__':
    train_word2id()
