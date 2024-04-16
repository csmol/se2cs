import pandas as pd
import numpy as np
import math
import pickle
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
import sys
import os

from pathlib import Path


from . import birch_bellwether_p_CFS as birch_bell
from . import birch
from . import utils

import sys
import traceback
import warnings
warnings.filterwarnings("ignore")

from ..config import Config

DEFAULT_CONFIG = Config()

# Birch Cluster Creator
def cluster_driver(df,print_tree = True, config=DEFAULT_CONFIG):
    X = df.apply(pd.to_numeric)
    cluster = birch.birch(branching_factor=config.branching_factor)
        #X.set_index('Project Name',inplace=True)
    cluster.fit(X)
    cluster_tree,max_depth = cluster.get_cluster_tree()
        #cluster_tree = cluster.model_adder(cluster_tree)
    if print_tree:
        cluster.show_clutser_tree()
    return cluster,cluster_tree,max_depth

def build_BIRCH(attr_df, config=DEFAULT_CONFIG):
    cluster,cluster_tree,_ = cluster_driver(attr_df, config)
    return cluster,cluster_tree

def get_clusters(data_source):
    if platform.system() == 'Darwin' or platform.system() == 'Linux':
        _dir = data_source + '/'
    else:
        _dir = data_source + '\\'

    clusters = [(join(_dir, f)) for f in listdir(_dir) if Path(join(_dir, f)).is_dir()]
    return clusters

def get_bellwether(df):
    bellwether = df.median_score.idxmin()
    return bellwether


def calculate_performance(data_source,clusters,path, goal,month, config=DEFAULT_CONFIG):
    df_train = pd.read_pickle(data_source + '/train_data.pkl')
    cluster,cluster_tree = build_BIRCH(df_train, config)
    cluster_ids = []
    cluster_structure = {}
    size = {}
    for key in cluster_tree:
        if cluster_tree[key].depth != None:
            cluster_ids.append(key)
            if cluster_tree[key].depth not in cluster_structure.keys():
                cluster_structure[cluster_tree[key].depth] = {}
            cluster_structure[cluster_tree[key].depth][key] = cluster_tree[key].parent_id
            size[key] = cluster_tree[key].size
    count = 0
    score = []
    score_med = []
    cluster_info = {}
    for cluster in clusters:
        if cluster in ['level_1','level_0']:
            continue
        perf_df = pd.read_pickle(data_source + str(cluster) + '/goal_' + str(goal) + '.pkl')
        avg_score = {}
        for s_project in perf_df.keys():
            if s_project not in avg_score.keys():
                avg_score[s_project] = []
            for d_project in perf_df[s_project].keys():
                avg_score[s_project].append(perf_df[s_project][d_project])
        perf_df = pd.DataFrame.from_dict(avg_score, orient = 'index')
        perf_df['median_score'] = perf_df.median(axis = 1)
        best_project = get_bellwether(perf_df) # get the best project that minimum median score
        score_med.append([cluster,best_project])
    score_df = pd.DataFrame(score_med, columns = ['id','bellwether'])
    score_df = score_df.sort_values('id')
    score_df.to_csv(data_source + '/bellwether_level_2.csv')
    level_1_bellwethers = {}

    # for cluster in cluster_structure[2].keys():
    #     if cluster_structure[2][cluster] not in level_1_bellwethers.keys():
    #         level_1_bellwethers[cluster_structure[2][cluster]] = []
    #     level_1_bellwethers[cluster_structure[2][cluster]].append(score_df[score_df['id'] == str(cluster)].bellwether.values.tolist()[0])
    
    for cluster in cluster_structure[1].keys():
        if cluster not in level_1_bellwethers.keys():
            level_1_bellwethers[cluster] = list(df_train.loc[cluster_tree[cluster].data_points].index)
    score_med = []
    
    data_path = Path(data_source + 'level_1')
    if not data_path.is_dir():
        os.makedirs(data_path)
    
    for key in  level_1_bellwethers.keys():
        sub_cluster_bellwethers = level_1_bellwethers[key]
        bell = birch_bell.Bellwether(path,df_train, goal,month, config=config)
        results = bell.bellwether(sub_cluster_bellwethers,sub_cluster_bellwethers)
        final_score = results[0]
        final_models = results[1]
        final_features = results[2]
        with open(data_source + 'level_1/cluster_'  + str(key) + '_performance.pkl', 'wb') as handle:
            pickle.dump(final_score, handle, protocol=pickle.HIGHEST_PROTOCOL)  
            
        with open(data_source + 'level_1/cluster_'  + str(key) + '_models.pkl', 'wb') as handle:
            pickle.dump(final_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(data_source + 'level_1/cluster_'  + str(key) + '_features.pkl', 'wb') as handle:
            pickle.dump(final_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    level_1_bellwethers = {}
    score_med = []
    for cluster in cluster_structure[1].keys():
        cluster_perf = pd.read_pickle(data_source + 'level_1/cluster_'  + str(cluster) + '_performance.pkl')
        level_1_avg_score = {}
        for s_project in cluster_perf.keys():
            if s_project not in level_1_avg_score.keys():
                level_1_avg_score[s_project] = []
            for d_project in cluster_perf[s_project].keys():
                level_1_avg_score[s_project].append(cluster_perf[s_project][d_project])
        cluster_perf_df = pd.DataFrame.from_dict(level_1_avg_score, orient = 'index')
        cluster_perf_df['median_score'] = cluster_perf_df.median(axis = 1)
        best_project = get_bellwether(cluster_perf_df)
        score_med.append([cluster,best_project])
    score_df = pd.DataFrame(score_med, columns = ['id','bellwether'])
    score_df = score_df.sort_values('id')
    score_df.to_csv(data_source + '/bellwether_level_1.csv')
    sub_cluster_bellwethers = score_df.bellwether.values.tolist()
    results = bell.bellwether(sub_cluster_bellwethers,sub_cluster_bellwethers)
    final_score = results[0]
    final_models = results[1]
    final_features = results[2]
    
    data_path = Path(data_source + 'level_0')
    if not data_path.is_dir():
        os.makedirs(data_path)
    
    with open(data_source + '/level_0/cluster_0_performance.pkl', 'wb') as handle:
        pickle.dump(final_score, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(data_source + '/level_0/cluster_0_models.pkl', 'wb') as handle:
        pickle.dump(final_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_source + '/level_0/cluster_0_features.pkl', 'wb') as handle:
        pickle.dump(final_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    level_0_avg_score = {}
    score_med = []
    for s_project in final_score.keys():
        if s_project not in level_0_avg_score.keys():
            level_0_avg_score[s_project] = []
        for d_project in final_score[s_project].keys():
            level_0_avg_score[s_project].append(final_score[s_project][d_project])  
    cluster_perf_df = pd.DataFrame.from_dict(level_0_avg_score, orient = 'index')
    cluster_perf_df['median_score'] = cluster_perf_df.median(axis = 1)
    best_project = get_bellwether(cluster_perf_df)
    score_med.append([0,best_project])
    score_df = pd.DataFrame(score_med, columns = ['id','bellwether'])
    score_df = score_df.sort_values('id')
    score_df.to_csv(data_source + '/bellwether_level_0.csv')


def find_bellwether(config=DEFAULT_CONFIG):
    month = config.month
    no_goals= config.no_goals
    seed = config.seed

    for i in range(no_goals):
        goal = utils.get_goal(i, config)
        path = "data/data_use/"
        data_source = 'results/with_CFS_DE/'+str(seed)+'/month_' + str(month) + '_models/' + goal + '/'
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            _dir = data_source + '/'
        else:
            _dir = data_source + '\\'
        clusters = [f for f in listdir(_dir) if Path(join(_dir, f)).is_dir()]
        calculate_performance(data_source,clusters,path,i,month, config)  


if __name__ == "__main__":
    month = 12
    no_goals= 7
    seeds={
    'seeds1-1':[1836],
    'seeds2-1':[8023],
    'seeds3-1':[6241],
    'seeds4-1':[3578],
    'seeds5-1':[5347],
    'seeds6-1':[2916],
    'seeds7-1':[1021],
    'seeds8-1':[9562],
    'seeds9-1':[8115],
    'seeds10-1':[6931],
    'seeds11-1':[2365],
    'seeds12-1':[9496],
    'seeds13-1':[6284],
    'seeds14-1':[4096],
    'seeds15-1':[5732],
    'seeds1-2':[4629],
    'seeds2-2':[9774],
    'seeds3-2':[1592],
    'seeds4-2':[8865],
    'seeds5-2':[7264],
    'seeds6-2':[6489],
    'seeds7-2':[3745],
    'seeds8-2':[5207],
    'seeds9-2':[2653],
    'seeds10-2':[4815],
    'seeds11-2':[7713],
    'seeds12-2':[3127],
    'seeds13-2':[1790],
    'seeds14-2':[9012],
    'seeds15-2':[1764],
    }
    seed_selected = sys.argv[1]
    seeds=seeds[seed_selected]
    for seed in seeds:
        for i in range(no_goals):
            goal = utils.get_goal(i, DEFAULT_CONFIG)
            path = 'data/data_use/'
            data_source = 'results/with_CFS_DE/'+str(seed)+'/month_' + str(month) + '_models/' + goal + '/'
            if platform.system() == 'Darwin' or platform.system() == 'Linux':
                _dir = data_source + '/'
            else:
                _dir = data_source + '\\'
            clusters = [f for f in listdir(_dir) if Path(join(_dir, f)).is_dir()]
            print("Clusters: ")
            print(clusters)
            calculate_performance(data_source,clusters,path,i,month, DEFAULT_CONFIG)  

