'''
main: 数据处理与分析过程，以及图嵌入(loc2vec/node2vec)/图神经网络(GATs/GCN)与时序模型(LSTM/GRU)结合的训练过程与效果评估
'''
from numpy.core.defchararray import index
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
from sklearn.metrics import roc_auc_score
import datetime
import folium
from folium import plugins
import webbrowser
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import cm 
import networkx as nx
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
import keras
from sklearn.cluster import DBSCAN, OPTICS, MeanShift
from GraphAttentionLayer import GraphAttention
from GraphConvolution import GraphConvolution
from tensorflow.keras.regularizers import l2
from sklearn.metrics import f1_score, precision_score, recall_score

## 全局变量
RESO = 9 # 或 8
EMBEDDING_SIZE = 128
L2V_COLUMNS = ['l2v_'+ str(x) for x in range(1,EMBEDDING_SIZE+1)]
N2V_COLUMNS = ['n2v_'+ str(x) for x in range(1,EMBEDDING_SIZE+1)]
FIELDS_COLUMNS = ['lat','lng','alt','month','m_day','hour', 'weekday', 'user_id'] + L2V_COLUMNS + N2V_COLUMNS
NUMERIC_COLUMNS = ['lat', 'lng', 'alt']
CATEGORICAL_COLUMNS = ['month','m_day','hour','weekday', 'user_id']
STEPS = 2 #或4，序列最长输入长度


# 六边形网格化
def trans_to_grid(df):
    # 城市h3网格化
    df['h3_grid'] = h3.geo_to_h3(df.lat, df.lng, resolution = RESO)
    df.lat, df.lng = h3.h3_to_geo(df.h3_grid)
    return df
def grid(group):
    global INTERVAL_ADJ_LS, INTERVAL_LS, LEN_LS
    path_id = group.iloc[0].path_id
    if path_id%500==0:
        print(path_id, end = '\t')
    group = group.apply(trans_to_grid, axis=1)
    group = group.drop_duplicates(['lat','lng'],keep='first')
    return group

# 筛去偏移点
def clear_path(df):
    adj_time_ls = []# 邻接两点时间间隔
    adj_dist_ls = []# 邻接两点距离
    path_len_ls = []# 路径点数
    se_time_ls = []# 起终点时间间隔
    se_dist_ls = []# 起终点距离
    error_path_ls = []#异常路径
    onepoint_path_ls = [] # 一个点
    del_index_ls = [] # 要删除的偏移点index
    for pathid, group in df.groupby('path_id'):
        if len(group)==1:
            onepoint_path_ls.append(pathid)
            continue
        se_time = group.iloc[-1].timestamp-group.iloc[0].timestamp
        se_dist = geodesic([group.iloc[0].lat, group.iloc[0].lng],
                     [group.iloc[-1].lat, group.iloc[-1].lng]).m
        if se_time>24*3600 or se_dist>80000:# 出北京市排除，时间范围超出排除
            error_path_ls.append(pathid)
            continue
        if se_time<120:
            onepoint_path_ls.append(pathid)
            continue
        flag = 0
        adj_time_ls_tmp = list(group.timestamp.values[1:]-group.timestamp.values[:-1])
        adj_time_ls_tmp2 = []
        adj_dist_ls_tmp = []
        del_index_ls_tmp = []
        tmp_grid = [group.iloc[0].lat, group.iloc[0].lat]
        tmp_interval = 0
        for i in range(len(group)-1):
            # 假设起点不偏移
            disttmp = geodesic(tmp_grid,
                     [group.iloc[i+1].lat, group.iloc[i+1].lat]).m
            tmp_interval += adj_time_ls_tmp[i]
            if disttmp>tmp_interval*28 or adj_time_ls_tmp[i]==0 or disttmp>80000:
                del_index_ls_tmp.append(group.index[i+1])
                continue
            tmp_grid = [group.iloc[i+1].lat, group.iloc[i+1].lat]
            tmp_interval = 0
            flag = 1
        if flag==1:
            se_time_ls.append(se_time)
            se_dist_ls.append(se_dist)
            del_index_ls += del_index_ls_tmp
            path_len_ls.append(len(group)-len(del_index_ls_tmp))
        else:
            error_path_ls.append(pathid)
    return {
           'path_len_ls':path_len_ls,'se_time_ls':se_time_ls,
           'se_dist_ls':se_dist_ls, 'del_index_ls':del_index_ls,
            'error_path_ls':error_path_ls,
           'onepoint_path_ls':onepoint_path_ls}

# 提取特征
def ext_fea(data_df_beijing_grid):
    global L2V_COLUMNS, N2V_COLUMNS
    # 注意
    data_df_beijing_grid = data_df_beijing_grid[data_df_beijing_grid.lat<41]
    data_df_beijing_grid = data_df_beijing_grid.reset_index(drop = True)
    # 修改海拔异常值
    data_df_beijing_grid.loc[data_df_beijing_grid.alt>1500,'alt']=1500
    data_df_beijing_grid.loc[data_df_beijing_grid.alt<0,'alt']=0
    # 提取年月日
    data_df_beijing_grid['datetime'] = data_df_beijing_grid['date']+' '+data_df_beijing_grid['time']
    data_df_beijing_grid['month'] = pd.to_datetime(data_df_beijing_grid['datetime']).dt.month
    data_df_beijing_grid['m_day'] = pd.to_datetime(data_df_beijing_grid['datetime']).dt.day
    data_df_beijing_grid['hour'] = pd.to_datetime(data_df_beijing_grid['datetime']).dt.hour
    data_df_beijing_grid['weekday'] = pd.to_datetime(data_df_beijing_grid['datetime']).dt.dayofweek
    # data_df_beijing_grid['minute'] = pd.to_datetime(data_df_beijing_grid['datetime']).dt.minute
    
    # l2v
    data_df_beijing_grid = pd.concat([data_df_beijing_grid, pd.DataFrame(columns=L2V_COLUMNS)])
    
    # n2v
    data_df_beijing_grid = pd.concat([data_df_beijing_grid, pd.DataFrame(columns=N2V_COLUMNS)])
    
    # stop point
    data_df_beijing_grid['staypoint']=['0']*len(data_df_beijing_grid)
    data_df_beijing_grid['start_ts']=[0.0]*len(data_df_beijing_grid)
    data_df_beijing_grid['end_ts']=[0.0]*len(data_df_beijing_grid)
    return data_df_beijing_grid

# 提取驻点
DIST_THRED = 1500 # 单位m
TIME_THRED = 600 # 单位s
def staypoint_detect(group):
    pathlen=len(group)
    if pathlen == 1:
        return group[group.staypoint!='0']
    if group.iloc[-1].timestamp-group.iloc[0].timestamp<=TIME_THRED:
        return group[group.staypoint!='0']
    i = 0
    # 起点
    group['start_ts'].iloc[i] = group.iloc[i].timestamp
    group['end_ts'].iloc[i] = group.iloc[i].timestamp
    group['staypoint'].iloc[i] = group.iloc[i].h3_grid
    while i < pathlen:
        j = i+1
        lat_acc = group.iloc[i].lat 
        lng_acc = group.iloc[i].lng
        while j < pathlen:
            # TODO: 
            tmpdist = geodesic([group.iloc[i].lat,group.iloc[i].lng],
                               [group.iloc[j].lat,group.iloc[j].lng]).m
            delta_t=group.iloc[j].timestamp-group.iloc[i].timestamp
            if tmpdist <= DIST_THRED:
                lat_acc+=group.iloc[j].lat
                lng_acc+=group.iloc[j].lng
            else:
                if delta_t>TIME_THRED:
                    stay_lat = lat_acc/(j-i)
                    stay_lng = lng_acc/(j-i)
                    group['staypoint'].iloc[i] = h3.geo_to_h3(stay_lat, stay_lng, resolution=RESO)
                    group['start_ts'].iloc[i] = group.iloc[i].timestamp
                    group['end_ts'].iloc[i] = group.iloc[j].timestamp
                # 注意
                i = j
                break
            j+=1
        if j>=pathlen:
            break
    # 终点
    group['staypoint'].iloc[-1] = group['h3_grid'].iloc[-1]
    group['start_ts'].iloc[-1] = group['timestamp'].iloc[-1]
    group['end_ts'].iloc[-1] = group['timestamp'].iloc[-1]
    
    if group.iloc[0].path_id%500==0:
        print(group.iloc[0].path_id, end = '\t')
    group = group[group.staypoint!='0'].drop_duplicates(['staypoint'],keep='first')
    return group

# 分析
DIST_THRELD = 80000
def analyse_nosp_path(df):
    adj_time_ls = []# 邻接两点时间间隔
    adj_dist_ls = []# 邻接两点距离
    path_len_ls = []# 路径点数
    se_time_ls = []# 起终点时间间隔
    se_dist_ls = []# 起终点距离
    error_path_ls = []#异常路径
    onepoint_path_ls = [] # 一个点
    del_index_ls = []
    for pathid, group in df.groupby('path_id'):
        if len(group)==1:
            onepoint_path_ls.append(pathid)
            continue
        se_time = group.iloc[-1].timestamp-group.iloc[0].timestamp
        se_dist = geodesic(list(h3.h3_to_geo(group.iloc[0].h3_grid)),
                     list(h3.h3_to_geo(group.iloc[-1].h3_grid))).m
        if se_time>24*3600 or se_dist>DIST_THRELD:# 出北京市排除，时间范围超出排除
            error_path_ls.append(pathid)
            continue
        if se_time<600:
            onepoint_path_ls.append(pathid)
            continue
        flag = 1
        adj_time_ls_tmp = list(group.timestamp.values[1:]-group.timestamp.values[:-1])
        adj_time_ls_tmp2 = []
        adj_dist_ls_tmp = []
        del_index_ls_tmp = []
        tmp_grid = group.iloc[0].h3_grid
        tmp_interval = 0
        for i in range(len(group)-1):
            # 假设起点不偏移
            disttmp = geodesic(list(h3.h3_to_geo(tmp_grid)),
                     list(h3.h3_to_geo(group.iloc[i+1].h3_grid))).m
            tmp_interval += adj_time_ls_tmp[i]
            if disttmp>tmp_interval*28 or adj_time_ls_tmp[i]==0 or tmp_interval<120 or disttmp > DIST_THRELD:
                del_index_ls_tmp.append(group.index[i+1])
                continue
            tmp_grid = group.iloc[i+1].h3_grid
            adj_time_ls_tmp2.append(tmp_interval)
            tmp_interval = 0
            adj_dist_ls_tmp.append(disttmp)
        if adj_dist_ls_tmp:
            se_time_ls.append(se_time)
            se_dist_ls.append(se_dist)
            adj_time_ls += adj_time_ls_tmp2
            adj_dist_ls += adj_dist_ls_tmp
            del_index_ls += del_index_ls_tmp
            path_len_ls.append(len(group)-len(del_index_ls_tmp))
        else:
            error_path_ls.append(pathid)
        if pathid%500==0:
            print(pathid,end = ' ')
    return {'adj_time_ls':adj_time_ls,'adj_dist_ls':adj_dist_ls,
           'path_len_ls':path_len_ls,'se_time_ls':se_time_ls,
           'se_dist_ls':se_dist_ls, 'del_index_ls':del_index_ls,
            'error_path_ls':error_path_ls,
           'onepoint_path_ls':onepoint_path_ls}

# 转换为聚类后的地点类别
def apply_cluster_grid(df):
    global encoder_dict, lables, cluster
    # -1 咋办
    label_tmp = lables[encoder_dict[df.h3_grid]]
    df['cluster_label'] = label_tmp
    df['cluster_lat'] = cluster.cluster_centers_[label_tmp][0]
    df['cluster_lng'] = cluster.cluster_centers_[label_tmp][1]
    df['cluster_grid'] = h3.geo_to_h3(df.cluster_lat,df.cluster_lng,resolution=RESO)
    return df

# 提取图嵌入后的节点向量
def ext_l2v_n2v(df):
    global l2v, n2v, L2V_COLUMNS, N2V_COLUMNS
    df[L2V_COLUMNS] = l2v[df.h3_grid]
    df[N2V_COLUMNS] = n2v[df.h3_grid]
    return df

# 返回的加入l2v或n2v或什么都不加的数据集
def return_data(data, cnt, mode = 'n2v', if_embedding = True):
    global NUMERIC_COLUMNS, CATEGORICAL_COLUMNS
    if not if_embedding:
        return data[:cnt,:,:len(NUMERIC_COLUMNS+CATEGORICAL_COLUMNS)]
    if mode=='n2v':
        return np.concatenate((data[:cnt,:,:len(NUMERIC_COLUMNS+CATEGORICAL_COLUMNS)], 
                               data[:cnt,:,-128:]),axis=-1)
    elif mode == 'l2v':
        return data[:cnt,:,:-128]

## 模型评估函数
def cal_auc(y_true, y_pred):
    auc_sum=0
    auc_cnt=0
    for i in range(y_true.shape[1]):
        try:
            auc_tmp = roc_auc_score(y_true[:,i],y_pred[:,i])
            auc_sum += auc_tmp
            auc_cnt += 1
        except:
            print(i, end='\t')
    return auc_sum/auc_cnt

def inverse_trans_top5(test_pred_y):
    top_5_idx = np.argsort(test_pred_y, axis=1)[:,-5:]# 顺序增长
    return top_5_idx

def label_acc(top_5, true_onehot):
    true_label = np.where(true_onehot==1)[1]
    top1_acc_cnt = 0
    top15_acc_cnt = 0
    
    ma_f1 = f1_score(top_5[:,-1], true_label, average='macro' )
    ma_p = precision_score(top_5[:,-1], true_label, average='macro')
    ma_r = recall_score(top_5[:,-1], true_label, average='macro')
    
    mi_f1 = f1_score(top_5[:,-1], true_label, average='micro' )
    mi_p = precision_score(top_5[:,-1], true_label, average='micro')
    mi_r = recall_score(top_5[:,-1], true_label, average='micro')
    
    w_f1 = f1_score(top_5[:,-1], true_label, average='weighted' )
    w_p = precision_score(top_5[:,-1], true_label, average='weighted')
    w_r = recall_score(top_5[:,-1], true_label, average='weighted')
    
    if true_label.shape[0]!=top_5.shape[0]:
        raise 'shape error'
    for i in range(true_label.shape[0]):
        if top_5[i,-1]== true_label[i]:
            top1_acc_cnt+=1
            top15_acc_cnt+=1
        elif true_label[i] in top_5[i]:
            top15_acc_cnt+=1
    return {'top1_acc':top1_acc_cnt/true_label.shape[0],
           'top5_acc':top15_acc_cnt/true_label.shape[0]
           ,'macro_p':ma_p
           ,'macro_r':ma_r
           ,'macro_f1':ma_f1
            ,'mi_p':mi_p
            ,'mi_r':mi_r
            ,'mi_f1':mi_f1
            ,'w_p':w_p
            ,'w_r':w_r
            ,'w_f1':w_f1
           }


if __name__=='__main__':
    '''
    数据预处理
    以GeoLife数据集为例
    '''
    path = '../data/ttl.csv'
    seed = 1 # 设置随机种子
    bandwidth = 0.07 # 决定聚类数量的参数
    data_df_beijing = pd.read_csv(path)
    data_df_beijing_grid_init = data_df_beijing.groupby(['path_id'],as_index=False).apply(grid)
    r9_clear = clear_path(data_df_beijing_grid_init)
    # 去除异常点 和 1个点的路径
    data_fea_df_nosp = data_df_beijing_grid_init.drop(index = r9_clear['del_index_ls'])
    data_fea_df_nosp = data_fea_df_nosp[~data_fea_df_nosp.path_id.isin(r9_clear['error_path_ls'])]
    data_fea_df_nosp = data_fea_df_nosp[~data_fea_df_nosp.path_id.isin(r9_clear['onepoint_path_ls'])]
    data_fea_df_nosp = data_fea_df_nosp.reset_index(drop=True)
    data_df_beijing_grid_9 = ext_fea(data_fea_df_nosp)
    df_beijing_sp_r9 = data_df_beijing_grid_9.groupby('path_id',as_index=False).apply(staypoint_detect)
    data_df_beijing_grid = df_beijing_sp_r9.reset_index(drop=True)
    data_df_beijing_grid_ana_res = analyse_nosp_path(data_df_beijing_grid)
    data_features_df = data_df_beijing_grid.drop(index = data_df_beijing_grid_ana_res['del_index_ls'])
    data_features_df = data_features_df[~data_features_df.path_id.isin(data_df_beijing_grid_ana_res['error_path_ls'])]
    data_features_df = data_features_df[~data_features_df.path_id.isin(data_df_beijing_grid_ana_res['onepoint_path_ls'])]
    data_features_df = data_features_df.reset_index(drop=True)
    data_features_df.to_csv('../data/data_features_df.csv', index=False)
    # 分割 train, validate, test
    random.seed(seed) # seed为随机种子
    path_num_tmp = np.max(data_features_df.path_id)
    pathid_ls = list(range(path_num_tmp))
    random.shuffle(pathid_ls)
    validate_path_id = pathid_ls[-int(path_num_tmp/8):]
    test_path_id = random.sample(pathid_ls[:-int(path_num_tmp/8)],int(path_num_tmp/5))
    print(len(validate_path_id),len(test_path_id))
    split_dataset = {}
    split_dataset['validate'] = validate_path_id
    split_dataset['test'] = test_path_id
    # 聚类
    # 所有node
    node_list = list(set(data_features_df[~data_features_df.isin(validate_path_id+test_path_id)].h3_grid))
    encoder_dict = {key:val for key, val in zip(node_list, list(range(len(node_list))))}
    node_lat_lng_array = np.zeros((len(node_list),2))
    for i in range(len(node_list)):
        node_lat_lng_array[i,0], node_lat_lng_array[i,1] = h3.h3_to_geo(node_list[i])
    db = MeanShift(bandwidth=bandwidth) 
    cluster = db.fit(node_lat_lng_array)
    lables = cluster.labels_
    # 将聚类后的类别加入
    data_features_df['cluster_grid'] = ['0']*len(data_features_df)
    data_features_df['cluster_label'] = [-1]*len(data_features_df)
    data_features_df['cluster_lat'] = [0]*len(data_features_df)
    data_features_df['cluster_lng'] = [0]*len(data_features_df)

    df_fea_cluster=data_features_df.apply(apply_cluster_grid,axis=1).drop_duplicates(['path_id','cluster_grid'],keep='first')
    df_fea_cluster['h3_grid'] = df_fea_cluster['cluster_grid']
    df_fea_cluster = df_fea_cluster.reset_index(drop=True)
    df_fea_cluster_ana = analyse_nosp_path(df_fea_cluster)
    # 剔除1个点的路径
    clear_df_fea_cluster = df_fea_cluster[~df_fea_cluster.path_id.isin(df_fea_cluster_ana['onepoint_path_ls'])]
    clear_df_fea_cluster = clear_df_fea_cluster.reset_index(drop=True)

    cluster_centers = cluster.cluster_centers_
    cluster_encoder_dict = {h3.geo_to_h3(cluster_centers[i,0], cluster_centers[i,1], resolution=RESO):i 
                            for i in range(cluster_centers.shape[0])}
    np.save('../data/cluster_encoder_dict.npy',cluster_encoder_dict)
    final_node_dict = cluster_encoder_dict
    final_node_list = ['0']*len(cluster_encoder_dict)
    for grid, no in cluster_encoder_dict.items():
        final_node_list[no] = grid

    final_node_lat_lng_array = np.zeros((len(final_node_list),2))
    for i in range(len(final_node_list)):
        final_node_lat_lng_array[i] = h3.h3_to_geo(final_node_list[i])
    # 构建图
    graph = nx.Graph()
    graph.add_nodes_from(final_node_list)
    for pathid, group in clear_df_fea_cluster[~clear_df_fea_cluster.path_id.isin(test_path_id+validate_path_id)].groupby('path_id'):
        path_tmp = group.h3_grid.values.tolist()
        if len(path_tmp)==1:
            graph.add_edge(path_tmp[0],path_tmp[0],weight = 1)
        for i in range(len(path_tmp)-1):
            graph.add_edge(path_tmp[i],path_tmp[i+1],weight = 1)
    adj = nx.adjacency_matrix(graph)
    A = np.array(adj.todense()) # 邻接矩阵

    # 转换成[[]]
    LOC_LL = [[]]*len(set(clear_df_fea_cluster[~clear_df_fea_cluster.path_id.isin(test_path_id+validate_path_id)].path_id))
    cnttmp = 0
    len_ls = []
    for _, group in clear_df_fea_cluster[~clear_df_fea_cluster.path_id.isin(test_path_id+validate_path_id)].groupby('path_id'):
        LOC_LL[cnttmp] = group.h3_grid.values.tolist()
        cnttmp+=1
    # embedding的参数，根据情况调节
    WINDOW_SIZE = 2
    MIN_COUNT = 1
    WALK_LENGTH = 12
    NUM_WALKS = 100
    # Loc2Vec
    l2v_model = Word2Vec(sentences=LOC_LL, size = EMBEDDING_SIZE, window = WINDOW_SIZE, min_count = MIN_COUNT
                        , sg = 1
                        , sample = 1e-3 # 高频词负采样
                        )
    l2v = l2v_model.wv
    # Node2Vec
    node2vec = Node2Vec(graph, dimensions=EMBEDDING_SIZE, walk_length=WALK_LENGTH, num_walks=NUM_WALKS, workers=2)
    node2vec_w2v = node2vec.fit(window = WINDOW_SIZE, min_count = MIN_COUNT
                            , sg = 1
                            , sample = 1e-3)
    n2v = node2vec_w2v.wv

    clear_df_fea_cluster = clear_df_fea_cluster.apply(ext_l2v_n2v, axis=1)
    clear_df_fea_cluster.to_csv('../data/clear_df_fea_cluster.csv', index=False)
    # 取极大极小值
    max_min_dict = {}
    for i, ft in enumerate(NUMERIC_COLUMNS+CATEGORICAL_COLUMNS):
        max_min_dict[ft] = [clear_df_fea_cluster[~clear_df_fea_cluster.path_id.isin(test_path_id+validate_path_id)][ft].max()
        , clear_df_fea_cluster[~clear_df_fea_cluster.path_id.isin(test_path_id+validate_path_id)][ft].min()]
    # 极大极小中心化
    n_data_features_df = clear_df_fea_cluster.copy()
    for key in max_min_dict.keys():
        maxtmp = max_min_dict[key][0]
        mintmp = max_min_dict[key][1]
        if key in NUMERIC_COLUMNS:
            n_data_features_df[key] = (n_data_features_df[key].values-mintmp)/(maxtmp-mintmp)
        else:
            n_data_features_df[key] = (n_data_features_df[key].values-mintmp)/(maxtmp-mintmp)
    n_final_node_lat_lng_array = final_node_lat_lng_array.copy()
    for i,key in enumerate(['lat','lng']):
        maxtmp = max_min_dict[key][0]
        mintmp = max_min_dict[key][1]
        n_final_node_lat_lng_array[:,i] = (n_final_node_lat_lng_array[:,i]-mintmp)/(maxtmp-mintmp)
    n_data_features_df = n_data_features_df[FIELDS_COLUMNS+['path_id', 'h3_grid']]

    # 生成训练、验证与测试集
    train_x = np.ones(shape=(100000,STEPS, len(FIELDS_COLUMNS)), dtype ='float32')*(-1000)
    train_y_onehot = np.ones(shape=(100000, len(final_node_dict)), dtype='float32')*(0)
    train_x_node = np.ones(shape=(100000,STEPS, len(final_node_dict)), dtype ='float32')*(0)

    validate_x  = np.ones(shape=(50000,STEPS, len(FIELDS_COLUMNS)), dtype ='float32')*(-1000)
    validate_y_onehot =  np.ones(shape=(50000, len(final_node_dict)), dtype='float32')*(0)
    validate_x_node = np.ones(shape=(50000,STEPS, len(final_node_dict)), dtype ='float32')*(0)

    test_x  = np.ones(shape=(50000,STEPS, len(FIELDS_COLUMNS)), dtype ='float32')*(-1000)
    test_y_onehot = np.ones(shape=(50000, len(final_node_dict)), dtype='float32')*(0)
    test_x_node = np.ones(shape=(50000,STEPS, len(final_node_dict)), dtype ='float32')*(0)

    train_cnt, test_cnt, validate_cnt = 0,0,0 

    for pathid, group in n_data_features_df.groupby('path_id'):
        pathlen = len(group)-1
        if pathlen==0:
            continue
        if pathid in validate_path_id:
            for i in range(pathlen, 0, -1):
                lentmp = random.randint(1,STEPS)
                # lentmp = STEPS
                validate_y_onehot[validate_cnt, final_node_dict[group.iloc[i]['h3_grid']]]=1
                validate_x[validate_cnt][:min(lentmp,i)]=group.iloc[max(0, (i-lentmp)):i][FIELDS_COLUMNS]

                for j in range(max(0, (i-lentmp)),i,1):
                    validate_x_node[validate_cnt, j-max(0, (i-lentmp)), final_node_dict[group.iloc[j]['h3_grid']]] = 1
                    
                validate_cnt+=1
        elif pathid in test_path_id:
            for i in range(pathlen, 0, -1):
                lentmp = random.randint(1,STEPS)
                # lentmp = STEPS
                test_y_onehot[test_cnt, final_node_dict[group.iloc[i]['h3_grid']]]=1
                test_x[test_cnt][:min(lentmp,i)]=group.iloc[max(0, (i-lentmp)):i][FIELDS_COLUMNS]
                
                for j in range(max(0, (i-lentmp)),i,1):
                    test_x_node[test_cnt, j-max(0, (i-lentmp)), final_node_dict[group.iloc[j]['h3_grid']]] = 1
                    
                test_cnt+=1
        else:
            for i in range(pathlen, 0, -1):
                lentmp = random.randint(1,STEPS)
                # lentmp = STEPS
                train_y_onehot[train_cnt, final_node_dict[group.iloc[i]['h3_grid']]]=1
                train_x[train_cnt][:min(lentmp,i)]=group.iloc[max(0, (i-lentmp)):i][FIELDS_COLUMNS]
                
                for j in range(max(0, (i-lentmp)),i,1):
                    train_x_node[train_cnt, j-max(0, (i-lentmp)), final_node_dict[group.iloc[j]['h3_grid']]] = 1
                    
                train_cnt+=1
        # if pathid%500==0:
        #     print(pathid, end='\t')


    '''
    搭建模型和训练
    '''
    # GRU
    K.clear_session()
    gru_model = Sequential()
    gru_model.add(Masking(mask_value=-1000.))
    gru_model.add(GRU(64, return_sequences = True, activation='relu'))
    gru_model.add(LayerNormalization())
    gru_model.add(GRU(32, activation='relu'))
    gru_model.add(LayerNormalization())
    gru_model.add(Dense(len(final_node_dict), activation='softmax'))
    log_dir = f'../model/log/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1
                                            ,update_freq = 'batch'
                                            )
    early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                verbose=1,
                patience=10,
                mode='max',
                restore_best_weights=True)
    callbacks = [early_stopping
                ,tbCallBack
                ]

    t1 = time.time()
    gru_model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy'])
    gru_model.fit(return_data(train_x, train_cnt, 'n2v',if_embedding=True), train_y_onehot[:train_cnt] # 选择n2v、l2v或是基线模型
                , batch_size=64
                , epochs= 30
                , shuffle=True,
                validation_data=(return_data(validate_x, validate_cnt, 'n2v', if_embedding=True), validate_y_onehot[:validate_cnt])
                ,callbacks=callbacks)
    t2 = time.time()
    print(t2-t1)

    # LSTM
    K.clear_session()
    lstm_model = Sequential()
    lstm_model.add(Masking(mask_value=-1000.))
    lstm_model.add(LSTM(64, return_sequences=True,activation='relu'))
    lstm_model.add(LayerNormalization())
    lstm_model.add(LSTM(32,activation='relu'))
    lstm_model.add(LayerNormalization())
    lstm_model.add(Dense(len(final_node_dict), activation='softmax'))
    log_dir = f'../model/log/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1
                                            ,update_freq = 'batch'
                                            )
    early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                verbose=1,
                patience=10,
                mode='max',
                restore_best_weights=True)
    callbacks = [early_stopping
                ,tbCallBack
                ]

    t1 = time.time()
    lstm_model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy'])
    lstm_model.fit(return_data(train_x, train_cnt, 'n2v', True), train_y_onehot[:train_cnt] # 选择n2v、l2v或是基线模型
                , batch_size=64
                , epochs= 30
                , shuffle=True,
                validation_data=(return_data(validate_x, validate_cnt, 'n2v', True), validate_y_onehot[:validate_cnt])
                ,callbacks=callbacks)
    t2 = time.time()
    print(t2-t1)

    final_node = final_node_lat_lng_array.copy() # final_node为轨迹点自身特征，可为经纬度，可继续适用node2vec或loc2vec
    # GATs-LSTM
    l2_reg = 5e-4/2
    F_ = final_node.shape[1] 
    Node_in = Input(shape=(STEPS, len(final_node_dict),))
    # 模型参数视情况调整
    graph_attention_1 = GraphAttention(F_,
                                    final_node,
                                    A,
                                    attn_heads=1,
                                    attn_heads_reduction='average',
                                    dropout_rate=0.5,
                                    activation='relu'
                                    #    ,kernel_regularizer=l2(l2_reg)
                                    #    ,attn_kernel_regularizer=l2(l2_reg)
                                    )([Node_in])

    inp_seq = Input((STEPS, len(FIELDS_COLUMNS)-128))
    mask_seq_inp = Masking(mask_value=-1000.)(inp_seq)
    mask_gat_inp = Masking(mask_value=0)(graph_attention_1)
    x = Concatenate()([mask_seq_inp,mask_gat_inp])

    x = LSTM(128, activation='relu', return_sequences=True)(x)
    x = LayerNormalization()(x)
    x = LSTM(64, activation='relu')(x)
    x = LayerNormalization()(x)
    out = Dense(len(final_node_dict),activation='softmax')(x)
    gat_lstm = Model([inp_seq, Node_in], out)

    log_dir = f'../model_checkin/log/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1
                                            ,update_freq = 'batch'
                                            )
    early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                verbose=1,
                patience=10,
                mode='max',
                restore_best_weights=True)
    callbacks = [early_stopping
                ,tbCallBack
                ]

    t1 = time.time()
    gat_lstm.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy'])
    gat_lstm.fit([return_data(train_x,train_cnt, mode='n2v', if_embedding=False)
                    , train_x_node[:train_cnt]]
                ,train_y_onehot[:train_cnt], 
                batch_size=64, epochs= 30, shuffle=True,
                validation_data=(
                    [return_data(validate_x, validate_cnt, mode='n2v', if_embedding=False)
                    , validate_x_node[:validate_cnt]]
                    , validate_y_onehot[:validate_cnt]
                )
                , callbacks = callbacks
                )
    t2 = time.time()
    print('time', t2-t1)

    # GCN-LSTM
    A1 = A + np.eye(A.shape[0])
    l2_reg = 5e-4/2
    F_ = final_node.shape[1]
    Node_in = Input(shape=(STEPS, len(final_node_dict),))

    D = np.zeros_like(A1)
    for i in range(A1.shape[0]):
        D[i,i] = np.power(np.sum(A1[i]),-1/2)

    features_adj = np.dot(np.dot(D,A1),D)

    gcn_1 = GraphConvolution(128,
                            features_adj,
                            activation='relu'
                        #    ,kernel_regularizer=l2(l2_reg)
                            )([Node_in])

    inp_seq = Input((STEPS, len(FIELDS_COLUMNS)-128))
    mask_seq_inp = Masking(mask_value=-1000.)(inp_seq)
    mask_gcn_inp = Masking(mask_value=0)(gcn_1)
    x = Concatenate()([mask_seq_inp,mask_gcn_inp])

    x = LSTM(128, activation='relu', return_sequences=True)(x)
    x = LayerNormalization()(x)
    x = LSTM(64, activation='relu')(x)
    x = LayerNormalization()(x)
    out = Dense(len(final_node_dict),activation='softmax')(x)
    gcn_lstm = Model([inp_seq, Node_in], out)

    log_dir = f'../model_checkin/log/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1
                                            ,update_freq = 'batch'
                                            )
    early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                verbose=1,
                patience=10,
                mode='max',
                restore_best_weights=True)
    callbacks = [early_stopping
                ,tbCallBack
                ]

    t1 = time.time()
    gcn_lstm.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy'])
    gcn_lstm.fit([return_data(train_x,train_cnt, mode='l2v', if_embedding=False)
                    , train_x_node[:train_cnt]]
                ,train_y_onehot[:train_cnt], 
                batch_size=64, epochs= 30, shuffle=True,
                validation_data=(
                    [return_data(validate_x, validate_cnt, mode='l2v', if_embedding=False)
                    , validate_x_node[:validate_cnt]]
                    , validate_y_onehot[:validate_cnt]
                )
                , callbacks = callbacks
                )
    t2 = time.time()
    print(t2-t1)

    '''
    模型评估
    '''
    # lstm/gru下选择n2v、l2v或是基线模型
    test_pred_y = gru_model.predict(return_data(test_x, test_cnt, 'n2v', True))
    print(gru_model.evaluate(return_data(test_x, test_cnt, 'n2v'), test_y_onehot[:test_cnt]))
    print(cal_auc(test_y_onehot[:test_cnt],test_pred_y[:test_cnt]))
    print(roc_auc_score(test_y_onehot[:test_cnt],test_pred_y[:test_cnt], average='micro'))

    test_pred_y = lstm_model.predict(return_data(test_x, test_cnt, 'n2v', True))
    print(lstm_model.evaluate(return_data(test_x, test_cnt, 'n2v'), test_y_onehot[:test_cnt]))
    print(cal_auc(test_y_onehot[:test_cnt],test_pred_y[:test_cnt]))
    print(roc_auc_score(test_y_onehot[:test_cnt],test_pred_y[:test_cnt], average='micro'))

    test_pred_y = gru_model.predict(return_data(test_x, test_cnt, 'n2v', True))
    train_pred_y = gru_model.predict(return_data(train_x, train_cnt, 'n2v', True))
    test_top_5_label = inverse_trans_top5(test_pred_y)
    train_top5_label = inverse_trans_top5(train_pred_y)
    test_label_res = label_acc(test_top_5_label, test_y_onehot[:test_cnt])
    train_label_res = label_acc(train_top5_label, train_y_onehot[:train_cnt])
    print(test_label_res,'\n',train_label_res)

    test_pred_y = lstm_model.predict(return_data(test_x, test_cnt, 'n2v', True))
    train_pred_y = lstm_model.predict(return_data(train_x, train_cnt, 'n2v', True))
    test_top_5_label = inverse_trans_top5(test_pred_y)
    train_top5_label = inverse_trans_top5(train_pred_y)
    test_label_res = label_acc(test_top_5_label, test_y_onehot[:test_cnt])
    train_label_res = label_acc(train_top5_label, train_y_onehot[:train_cnt])
    print(test_label_res,'\n',train_label_res)

    # GATs-LSTM
    gat_lstm.evaluate(
        [return_data(test_x, test_cnt, 'n2v', False)
        ,test_x_node[:test_cnt]]
        , test_y_onehot[:test_cnt])

    test_pred_y = gat_lstm.predict(
        [return_data(test_x, test_cnt, 'n2v', False)
        ,test_x_node[:test_cnt]]
        )
    train_pred_y = gat_lstm.predict(
        [return_data(train_x, train_cnt, 'n2v', False)
        ,train_x_node[:train_cnt]]
        )

    test_top_5_label = inverse_trans_top5(test_pred_y)
    train_top5_label = inverse_trans_top5(train_pred_y)

    test_label_res = label_acc(test_top_5_label, test_y_onehot[:test_cnt])
    train_label_res = label_acc(train_top5_label, train_y_onehot[:train_cnt])

    print(test_label_res,'\n',train_label_res)
    print(cal_auc(test_y_onehot[:test_cnt],test_pred_y[:test_cnt]))
    print(roc_auc_score(test_y_onehot[:test_cnt],test_pred_y[:test_cnt], average='micro'))

    # GCN-LSTM
    gcn_lstm.evaluate(
        [return_data(test_x, test_cnt, 'n2v', False)
        ,test_x_node[:test_cnt]]
        , test_y_onehot[:test_cnt])

    test_pred_y = gcn_lstm.predict(
        [return_data(test_x, test_cnt, 'l2v', False)
        ,test_x_node[:test_cnt]]
        )
    train_pred_y = gcn_lstm.predict(
        [return_data(train_x, train_cnt, 'l2v', False)
        ,train_x_node[:train_cnt]]
        )

    test_top_5_label = inverse_trans_top5(test_pred_y)
    train_top5_label = inverse_trans_top5(train_pred_y)
    test_label_res = label_acc(test_top_5_label, test_y_onehot[:test_cnt])
    train_label_res = label_acc(train_top5_label, train_y_onehot[:train_cnt])

    print(test_label_res,'\n',train_label_res)
    print(cal_auc(test_y_onehot[:test_cnt],test_pred_y[:test_cnt]))
    print(roc_auc_score(test_y_onehot[:test_cnt],test_pred_y[:test_cnt], average='micro'))
