# -*- coding:utf-8 -*- 

import pandas as pd
import pickle
from itertools import chain


def question_and_topic_2id():
    """把 question 和 knowledge 转成id形式并保存至 ../data/目录下。"""
    
    # question_id topic_id1,topic_id2...topic_idn
    print('Changing the quetion and knowledge to id and save in sr_question2.pkl and sr_knowledge2id.pkl in ../data/')
    df_question_topic = pd.read_csv('../raw_data/question_knowledge_train_set.txt', sep='\t', names=['question', 'topics'],
                        dtype={'question': object, 'topics': object})
    save_path = '../data/'
    print('questino number = %d ' % len(df_question_topic))
    
    # 问题 id 按照给出的问题顺序编号
    questions = df_question_topic.question.values
    sr_question2id = pd.Series(range(len(questions)), index=questions) 
    sr_id2question = pd.Series(questions, index=range(len(questions)))
    
    
    df_question_topic.topics = df_question_topic.topics.apply(lambda tps: tps.split(','))    
    topics = df_question_topic.topics.values
    # 高效的迭代器chain http://funhacks.net/2017/02/13/itertools/
    topics = list(chain(*topics)) 
    sr_topics = pd.Series(topics)
    # knowledge 按照数量从大到小进行编号
    topics_count = sr_topics.value_counts()
    topics = topics_count.index
    sr_topic2id = pd.Series(range(len(topics)),index=topics)
    sr_id2topic = pd.Series(topics, index=range(len(topics))) 

    with open(save_path + 'sr_question2id.pkl', 'wb') as outp:
        pickle.dump(sr_question2id, outp)
        pickle.dump(sr_id2question, outp)
    with open(save_path + 'sr_topic2id.pkl', 'wb') as outp:
        pickle.dump(sr_topic2id, outp)
        pickle.dump(sr_id2topic, outp)
    print('Finished changing.')


if __name__ == '__main__':
    question_and_topic_2id()
