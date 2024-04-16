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
from sklearn import tree

import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
import sys
import os
import copy
import traceback
from pathlib import Path

import traceback
import matplotlib.pyplot as plt

from . import birch
from .predictor_advance_v1 import *
from . import utils

import sys
import traceback
import warnings
warnings.filterwarnings("ignore")

from ..config import Config

DEFAULT_CONFIG = Config()

def prepare_data(repo_name, directory, goal, config=DEFAULT_CONFIG):
    df_raw = pd.read_csv(directory + repo_name, sep=',')
    df_raw = df_raw.drop(columns=['dates'])
    last_col = utils.get_goal(goal, config)
    cols = list(df_raw.columns.values)
    cols.pop(cols.index(last_col))
    df_adjust = df_raw[cols+[last_col]]
    return df_adjust

# Cluster Driver
def cluster_driver(df,goal,print_tree = True, config=DEFAULT_CONFIG):
    X = df.apply(pd.to_numeric)
    cluster = birch.birch(branching_factor=config.branching_factor)
    cluster.fit(X)
    cluster_tree,max_depth = cluster.get_cluster_tree()
    if print_tree:
        cluster.show_clutser_tree()
    return cluster,cluster_tree,max_depth

def get_predicted(cluster_data, project_data, goal_num, metrics, month,seed, config=DEFAULT_CONFIG):
    try:
        train_data = pd.read_pickle(cluster_data + '/train_data.pkl')
        
        cluster,cluster_tree,max_depth = cluster_driver(train_data,goal_num,config)
        goal = get_goal(goal_num, config)
        path  = 'results/with_CFS_DE/'+str(seed)+'/month_' + str(month) + '_models/' + goal
        print("GOAL NO: ",goal)
        conv_bell_df = pd.read_pickle('results/with_CFS_DE/'+str(seed)+'/month_' + str(month) + '_models/' + goal + '/default_bellwether.pkl')
        conv_bell_df = pd.DataFrame.from_dict(conv_bell_df, orient = 'index')
        # print("conv bell", conv_bell_df)
        conv_bell = conv_bell_df.median(axis = 0).idxmin()
        conv_bell_model_df = pd.read_pickle('results/with_CFS_DE/'+str(seed)+'/month_' + str(month) + '_models/' + goal + '/default_bellwether_models.pkl')
        conv_bell_clf = conv_bell_model_df[conv_bell]
        t_df = pd.DataFrame()
        for source_project in train_data.index:
            source_project_dataset = prepare_data(source_project, 
                                                    project_data, 
                                                    goal_num,
                                                    config)
            source_project_dataset = normalize(source_project_dataset)
            t_df = pd.concat([t_df,source_project_dataset])
        clf_global = DECART_test(t_df,metrics,month)
        
        # levels = [0,1,2] #TODO uncomment
        levels=[0,1]
        results = []
        for level in levels:
            bellwether_models = {}
            
            test_data = pd.read_pickle(cluster_data + '/test_data.pkl')
            
            bell_df = pd.read_csv(cluster_data + '/bellwether_level_' + str(level) + '.csv')
            bell_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
            
            predicted_clusters = cluster.predict(test_data,level)
            
            test_data['predicted_clusters'] = predicted_clusters
            
            for test_project in test_data.index:
                result = []
                result.append(test_project)
                result.append(level)
                
                predicted_cluster = test_data.loc[test_project,'predicted_clusters']
                # print("predicted clusters: ",predicted_cluster)
                # print("bell df", bell_df)
                predicted_cluster_bellwether = bell_df[bell_df['id'] == predicted_cluster].bellwether.values[0]
                
                if predicted_cluster_bellwether not in bellwether_models.keys():
                    bellwether_models[predicted_cluster_bellwether] = None
                
                testing_project_dataset = prepare_data(test_project, 
                                                    project_data, 
                                                    goal_num,
                                                    config)
                testing_project_dataset = normalize(testing_project_dataset)


                trainData, testData = df_split_month(testing_project_dataset, month)
                
                y_train = trainData[goal]
                X_train = trainData.drop([goal], axis = 1)
                
                y_test = testData[goal]
                X_test = testData.drop([goal], axis = 1)
                
                train_df = X_train
                train_df[goal] = y_train
                
                
                
                clf_test = DECART_test(train_df,metrics,month)
                # print("XTEST:  ")
                # print( X_test)
                # if not X_test:
                #     continue
                test_predict = clf_test.predict(X_test)
                test_actual = y_test.values
                
                mre = mre_calc(test_predict, test_actual)
                sa = sa_calc(test_predict, test_actual, y_train)
                
                result.append(mre)
                result.append(sa)

                bellwether_dataset = prepare_data(predicted_cluster_bellwether, 
                                                project_data, 
                                                goal_num,
                                                config)
                if level == 2:
                    with open(path + '/' + str(predicted_cluster) + '/goal_' +  str(goal_num) + '_models.pkl', 'rb') as handle:
                        models = pickle.load(handle)
                    with open(path + '/' + str(predicted_cluster) + '/goal_' +  str(goal_num) + '_features.pkl', 'rb') as handle:
                        features = pickle.load(handle)
                else:
                    with open(path + '/level_' + str(level) + '/cluster_' +  str(predicted_cluster) + '_models.pkl', 'rb') as handle:
                        models = pickle.load(handle)
                    with open(path + '/level_' + str(level) + '/cluster_' +  str(predicted_cluster) + '_features.pkl', 'rb') as handle:
                        features = pickle.load(handle)
                
                print(level, predicted_cluster_bellwether, predicted_cluster)
                clf_bellwether = models[predicted_cluster_bellwether]
                clf_bellwether_cols = features[predicted_cluster_bellwether][:-1]
                
                test_predict = clf_bellwether.predict(X_test[clf_bellwether_cols])
                test_actual = y_test.values
                
                mre = mre_calc(test_predict, test_actual)
                sa = sa_calc(test_predict, test_actual, y_train)
                
                result.append(mre)
                result.append(sa)

                test_predict = conv_bell_clf.predict(X_test)
                test_actual = y_test.values
                
                mre = mre_calc(test_predict, test_actual)
                sa = sa_calc(test_predict, test_actual, y_train)

                result.append(mre)
                result.append(sa)
        
                test_predict = clf_global.predict(X_test)
                test_actual = y_test.values
                
                mre = mre_calc(test_predict, test_actual)
                sa = sa_calc(test_predict, test_actual, y_train)

                result.append(mre)
                result.append(sa)
            
                results.append(result)
        
        results_df = pd.DataFrame(results, columns = ['project','level','self_mre','self_sa','bellwether_mre','bellwether_sa','conv_bell_mre','conv_bell_sa','global_mre','global_sa'])
        # results_df = pd.DataFrame(results, columns = ['project','level','bellwether_mre','bellwether_sa']) #todo add only bellwethers
                
        return results_df 
    except Exception as e:
        # Handle the exception here
        print("An error occurred in get_predicted:", e)
        traceback.print_exc()


def performance_calculator(config=DEFAULT_CONFIG):
    month = config.month
    no_goals = config.no_goals
    seed = config.seed

    for i in range(no_goals):
        goal = get_goal(i, config)
        path  = 'results/with_CFS_DE/'+str(seed)+'/month_' + str(month) + '_models/' + goal
        results_df = get_predicted(cluster_data = path, 
                    project_data = "data/data_use/", 
                    goal_num = i, 
                    metrics = 0, 
                    month = month,
                    seed= seed,
                    config=config)
        results_df.to_csv(path + '/results_tuned_final_sa.csv')


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
            goal = get_goal(i, DEFAULT_CONFIG)
            print("goal:", goal)
            path  = 'results/with_CFS_DE/'+str(seed)+'/month_' + str(month) + '_models/' + goal
            results_df = get_predicted(cluster_data = path, 
                        project_data = 'data/data_use/', 
                        goal_num = i, 
                        metrics = 0, 
                        month = month,
                        seed= seed,
                        config=DEFAULT_CONFIG)
            results_df.to_csv(path + '/results_tuned_final_sa.csv')