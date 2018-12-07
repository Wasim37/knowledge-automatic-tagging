# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division

from tqdm import tqdm
from bs4 import BeautifulSoup
from flask import jsonify
from flask import Flask
from flask import request

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
import time
import network
import jieba
import pickle
import json
import math

sys.path.append('../../')
from data_helpers import pad_X180
from evaluator import score_eval

settings = network.Settings()
title_len = settings.title_len
model_name = settings.model_name
ckpt_path = settings.ckpt_path
embedding_path = '../../data/word_embedding.npy'


save_path = '../../data/'
with open(save_path + 'sr_topic2id.pkl', 'rb') as inp2:
    sr_topic2id = pickle.load(inp2)
    sr_id2topic = pickle.load(inp2)
with open(save_path + 'sr_word2id.pkl', 'rb') as inp:
    sr_id2word = pickle.load(inp)
    sr_word2id = pickle.load(inp)
dict_word2id = dict()
for i in range(len(sr_word2id)):
    dict_word2id[sr_word2id.index[i]] = sr_word2id.values[i]


def get_id4words(words):
    """把 words 转为 对应的 id"""
    cut = jieba.cut(words.strip()) # jieba分词
    words = ','.join(cut).split(',') 
    # python3种 map()和filter()返回的不是list，如果要转换为列表，可以使用list()转换
    # https://blog.csdn.net/qq_42397914/article/details/81586060    
    ids = list(map(get_id, words))  # 获取id
    return ids


def get_id(word):
    """获取 word 所对应的 id.
    如果该词不在词典中，用 <UNK>（对应的 ID 为 1 ）进行替换。
    """
    if word not in dict_word2id:
        return 1
    else:
        return dict_word2id[word]

    
class knowledge_info(object):
    def __init__(self,node_id , node_name, score):
        self.node_id = node_id
        self.node_name = node_name
        self.score = score


def knowledge_info_2_json(obj):
    return {
        'node_id': obj.node_id,
        'node_name': obj.node_name,
        'score': obj.score
    }


def local_predict(sess, model, content):
    """知识点预测"""
    ids = np.asarray(list(map(get_id4words, np.asarray(content.split(), dtype=object))))
    X_batch = np.asarray(list(map(pad_X180, ids)))
    X1_batch = X_batch[:, :title_len]
    X2_batch = X_batch[:, title_len:]    
    _batch_size = len(X1_batch)
    fetches = [model.y_pred]
    feed_dict = {model.X1_inputs: X1_batch, model.X2_inputs: X2_batch,
                 model.batch_size: _batch_size, model.tst: True, model.keep_prob: 1.0}
    predict_labels = sess.run(fetches, feed_dict)[0] 
    return predict_labels


if not os.path.exists(ckpt_path + 'checkpoint'):
    print('there is not saved model, please check the ckpt path')
    exit()


#print('加载知识点节点id与名称......')
df_train = pd.read_csv('../../raw_data/all_knowledge_set.txt', sep='\t', usecols=[0, 1],
                        names=['topic_id', 'topic_name'], dtype={'topic_id': object})
dict_topic_id2name = dict(zip(df_train.topic_id, df_train.topic_name.values))


print('加载词向量......')
W_embedding = np.load(embedding_path)
print('定义模型结构......')
model = network.TextCNN(W_embedding, settings)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
print('初始化模型参数......')
model.saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))


app = Flask(__name__)
@app.route('/', methods=['POST','GET'])
def get_text_input():
    content = request.args.get('title')
    if not content:   
        return '参数有误，正确格式 http://127.0.0.1:5002/?title=输入文本'
    
    content = BeautifulSoup(content, "html.parser").text
    if len(content) < 10:
        return '文本去除公式图片等标签后，至少10个字符，请重新输入'
    
    # 预测得分
    predict_labels = local_predict(sess, model, content) 
    
    # 计算置信度
    y_exp = [math.exp(i) for i in predict_labels[0]]  
    sum_y_exp = sum(y_exp)  
    softmax = [round(i / sum_y_exp, 3) for i in y_exp]
    
    # 排序获取得分最高的节点
    predict_labels_top = np.argsort(-predict_labels[0])[:10]
    predict_labels_list = list()
    predict_labels_list.extend(predict_labels_top)
    print(predict_labels_list)
    
    # 拼装知识节点对象并返回
    knowledge_info_list = []
    for lable in predict_labels_top:
        node_id = sr_id2topic[lable]
        node_name = dict_topic_id2name[node_id]
        score = softmax[lable]
        knowledge_info_list.append(knowledge_info(node_id, node_name, score))
    return json.dumps(knowledge_info_list, default=knowledge_info_2_json)


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(host='127.0.0.1',port=5002)
    #app.run()
