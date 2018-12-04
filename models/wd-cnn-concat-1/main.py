# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division

from tqdm import tqdm
from flask import jsonify
from flask import Flask
from flask import request

import tensorflow as tf
import numpy as np
import os
import sys
import time
import network
import jieba
import pickle

sys.path.append('../../')
from data_helpers import pad_X180
from evaluator import score_eval

settings = network.Settings()
title_len = settings.title_len
model_name = settings.model_name
ckpt_path = settings.ckpt_path

local_scores_path = '../../local_scores/'
scores_path = '../../scores/'
if not os.path.exists(local_scores_path):
    os.makedirs(local_scores_path)
if not os.path.exists(scores_path):
    os.makedirs(scores_path)

embedding_path = '../../data/word_embedding.npy'
data_valid_path = '../../data/wd-data/data_valid/'
data_test_path = '../../data/wd-data/data_test/'
va_batches = os.listdir(data_valid_path)
te_batches = os.listdir(data_test_path)  # batch 文件名列表
n_va_batches = len(va_batches)
n_te_batches = len(te_batches)

#---------------

save_path = '../../data/'
with open(save_path + 'sr_word2id.pkl', 'rb') as inp:
    sr_id2word = pickle.load(inp)
    sr_word2id = pickle.load(inp)
dict_word2id = dict()
for i in range(len(sr_word2id)):
    dict_word2id[sr_word2id.index[i]] = sr_word2id.values[i]
    
save_path = '../../data/'
with open(save_path + 'sr_topic2id.pkl', 'rb') as inp2:
    sr_topic2id = pickle.load(inp2)
    sr_id2topic = pickle.load(inp2)

print("sss")
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

#---------------
    

def local_predict(sess, model, content):
    """将输入的文本转为模型可运行的数据"""
    ids = np.asarray(list(map(get_id4words, np.asarray(content.split(), dtype=object))))
    X_batch = np.asarray(list(map(pad_X180, ids)))
    print(X_batch)

    X1_batch = X_batch[:, :title_len]
    X2_batch = X_batch[:, title_len:]    

    _batch_size = len(X1_batch)
    fetches = [model.y_pred]
    feed_dict = {model.X1_inputs: X1_batch, model.X2_inputs: X2_batch,
                 model.batch_size: _batch_size, model.tst: True, model.keep_prob: 1.0}
    predict_labels = sess.run(fetches, feed_dict)[0] 
    return predict_labels


app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def get_text_input():
    content = request.args.get('title')
    if not content:   
        return '参数有误，正确格式 http://127.0.0.1:5002/?title=此处输入纯文本'
    
    if not os.path.exists(ckpt_path + 'checkpoint'):
        print('there is not saved model, please check the ckpt path')
        exit()
    print('Loading model...')
    W_embedding = np.load(embedding_path)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = network.TextCNN(W_embedding, settings)
        model.saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        print('Local predicting...')
        predict_labels = local_predict(sess, model, content)
        predict_labels = map(lambda label: label.argsort()[-1:-6:-1], predict_labels)
        predict_labels_list = list()
        predict_labels_list.extend(predict_labels)
        print(predict_labels_list)
        topc_id = [sr_id2topic[x] for x in predict_labels_list]
        print(topc_id)
       
    
    return "as"

if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(host='127.0.0.1',port=5002)
