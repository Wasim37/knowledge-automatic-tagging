# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pickle
import jieba
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


def topicid2name():
    """将对应学段学科的所有知识点id与name进行存储"""
    time0 = time.time()
    print('Processing data...')
    df_train = pd.read_csv('../raw_data/all_knowledge_set.txt', sep='\t', usecols=[0, 1],
                            names=['topic_id', 'topic_name'], dtype={'topic_id': object})
    
    print('knowledge number %d ' % len(df_train))  
    
    # 转为 id 形式
    topic_id2name = pd.Series(df_train.topic_name.values, index=df_train.topic_id)
    dict_topic_id2name = dict(zip(df_train.topic_id, df_train.topic_name.values))
    print(dict_topic_id2name)
    #np.savez('../data/topic_id2name.npz', topic_id2name)
    #train_content = np.load('../data/topic_id2name.npz')
    #print('Finished. Costed time %g s' % (time.time() - time0))


if __name__ == '__main__':
    topicid2name()
