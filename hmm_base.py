'''
基准模型HMM的训练与评估
'''
import numpy as np
import hmmlearn.hmm as hmm

import pandas as pd
import numpy as np
import os
import time
import h3
from gensim.models import Word2Vec
from collections import Counter
import copy
from geopy.distance import geodesic
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import random
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from node2vec import Node2Vec
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.layers import concatenate

# 评估模型效果
def evaluate(transmat, emissionprob, state, true_y):
    prob_mat = np.dot(transmat[state], emissionprob)
    top5_mat = np.argsort(prob_mat, axis=1)[:,-5:]# 顺序增长
    acc_cnt, acc_top5_cnt = 0, 0

    ma_f1 = f1_score(top5_mat[:,-1], true_y, average='macro' )
    ma_p = precision_score(top5_mat[:,-1], true_y, average='macro')
    ma_r = recall_score(top5_mat[:,-1], true_y, average='macro')

    mi_f1 = f1_score(top5_mat[:,-1], true_y, average='micro' )
    # print(mi_f1)
    mi_p = precision_score(top5_mat[:,-1], true_y, average='micro')
    # print(mi_p)
    mi_r = recall_score(top5_mat[:,-1], true_y, average='micro')
    # print(mi_r)

    w_f1 = f1_score(top5_mat[:,-1], true_y, average='weighted' )
    # print(mi_f1)
    w_p = precision_score(top5_mat[:,-1], true_y, average='weighted')
    # print(mi_p)
    w_r = recall_score(top5_mat[:,-1], true_y, average='weighted')
    # print(mi_r)

    for i in range(prob_mat.shape[0]):
        if top5_mat[i,-1]==true_y[i]:
            acc_cnt += 1
            acc_top5_cnt += 1
        elif true_y[i] in top5_mat[i]:
            acc_top5_cnt += 1
    return {'acc':acc_cnt/prob_mat.shape[0]
    , 'top5_acc':acc_top5_cnt/prob_mat.shape[0]
    , 'macro_p':ma_p
    , 'macro_r':ma_r
    , 'macro_f1':ma_f1
    , 'micro_p': mi_p
    , 'micro_r': mi_r
    , 'micro_f1': mi_f1
    , 'weighted_p': w_p
    , 'weighted_r': w_r
    , 'weighted_f1': w_f1
    }

if __name__=='__main__':
    '''
    以GeoLife为例
    '''
    RESO = 9
    seed = 1 # 设置随机种子

    clear_df_fea_cluster = pd.read_csv('../data/clear_df_fea_cluster.csv')
    data_features_df = pd.read_csv('../data/data_features_df.csv')
    cluster_encoder_dict = np.load('/cluster_encoder_dict.npy', allow_pickle=True).item()
    # 网格：idx，idx：网格
    final_node_dict = cluster_encoder_dict
    final_node_list = ['0']*len(cluster_encoder_dict)
    for grid, no in cluster_encoder_dict.items():
        final_node_list[no] = grid

    # 分割 train, validate, test
    random.seed(seed)
    path_num_tmp = np.max(data_features_df.path_id)
    pathid_ls = list(range(path_num_tmp))
    random.shuffle(pathid_ls)
    validate_path_id = pathid_ls[-int(path_num_tmp/8):]
    test_path_id = random.sample(pathid_ls[:-int(path_num_tmp/8)],int(path_num_tmp/5))
    print(len(validate_path_id),len(test_path_id))
    split_dataset = {}
    split_dataset['validate'] = validate_path_id
    split_dataset['test'] = test_path_id

    # 生成数据集
    train_x = np.ones(shape=(100000,1), dtype ='int64')*(-1000)

    validate_x  = np.ones(shape=(50000,1), dtype ='int64')*(-1000)
    validate_y = np.ones(shape=(50000,1), dtype ='int64')*(-1000)

    test_x  = np.ones(shape=(50000,1), dtype ='int64')*(-1000)
    test_y = np.ones(shape=(50000,1), dtype ='int64')*(-1000)

    train_cnt, test_cnt, validate_cnt = 0,0,0 
    cum_train_cnt, cum_test_cnt, cum_validate_cnt = 0,0,0

    train_len_ls, test_len_ls, validate_len_ls = [], [], []
    for pathid, group in clear_df_fea_cluster.groupby('path_id'):
        pathlen = len(group)
        if pathlen==1:
            continue
        if pathid in validate_path_id:
            validate_cnt += 1
            validate_x[cum_validate_cnt:(cum_validate_cnt+pathlen-1)]=group.cluster_label.values.reshape(-1,1)[:-1]
            validate_y[cum_validate_cnt:(cum_validate_cnt+pathlen-1)]=group.cluster_label.values.reshape(-1,1)[1:]
            cum_validate_cnt += (pathlen-1)
            validate_len_ls.append(pathlen-1)
        elif pathid in test_path_id:
            test_cnt+=1
            test_x[cum_test_cnt:(cum_test_cnt+pathlen-1)]=group.cluster_label.values.reshape(-1,1)[:-1]
            test_y[cum_test_cnt:(cum_test_cnt+pathlen-1)]=group.cluster_label.values.reshape(-1,1)[1:]
            cum_test_cnt += (pathlen-1)
            test_len_ls.append(pathlen-1)
        else:   
            train_cnt+=1
            train_x[cum_train_cnt:(cum_train_cnt+pathlen)]=group.cluster_label.values.reshape(-1,1)
            cum_train_cnt += pathlen
            train_len_ls.append(pathlen)
        if pathid%500==0:
            print(pathid, end='\t')

    print(train_cnt, test_cnt, validate_cnt)
    print(cum_train_cnt, cum_test_cnt, cum_validate_cnt)


    start_t = time.time()
    model = hmm.MultinomialHMM(n_components=64, n_iter=200, tol=0.000001) # 32,64
    model.fit(train_x[:cum_train_cnt],lengths=train_len_ls)
    end_t = time.time()
    print(end_t-start_t)
    test_state = model.predict(test_x[:cum_test_cnt],lengths=test_len_ls)
    test_acc_res = evaluate(model.transmat_, model.emissionprob_, test_state, test_y[:cum_test_cnt])
    print(test_acc_res)
    print(model.startprob_)
    print(model.transmat_.shape)
    print(model.emissionprob_.shape)
