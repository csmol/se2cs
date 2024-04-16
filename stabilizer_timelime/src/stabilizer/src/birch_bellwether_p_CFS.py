import pandas as pd
import numpy as np
import math
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
import sys
import os
import copy
import traceback
import timeit
import random


import matplotlib.pyplot as plt

from . import birch
from .predictor_advance_v1 import *
from . import utils
from . import CFS_regression as CFS


from multiprocessing import Pool, cpu_count
from threading import Thread
from multiprocessing import Queue

# import metrices

import sys
import traceback
import warnings
warnings.filterwarnings("ignore")

from ..config import Config

DEFAULT_CONFIG = Config()

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        #print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return



class Bellwether(object):

    def __init__(self,data_path,attr_df, goal, month, config=DEFAULT_CONFIG):
        self.directory = data_path
        self.attr_df = attr_df
        self.cores = 8
        self.goal = goal
        self.metrics = 0
        self.month = month
        self.config = config

            
    def prepare_data(self, repo_name):
        df_raw = pd.read_csv(self.directory + repo_name, sep=',')
        df_raw = df_raw.drop(columns=['dates'])  
        last_col = utils.get_goal(self.goal, self.config)
        cols = list(df_raw.columns.values)
        cols.sort()
        cols.remove(last_col)
        df_adjust = df_raw[cols+[last_col]]
        return df_adjust

    def apply_cfs(self, df):
        goal = utils.get_goal(self.goal, self.config)
        y = df[goal].values
        X_df = df.drop(labels = [goal],axis = 1)
        X = X_df.values
        selected_cols = CFS.cfs(X,y)
        cols = X_df.columns[[selected_cols]].tolist()[0]
        cols.append(goal)

        return df[cols],cols


    # Cluster Driver
    def cluster_driver(self,df,print_tree = True):
        X = df.apply(pd.to_numeric)
        cluster = birch.birch(branching_factor=self.config.branching_factor)
        cluster.fit(X)
        cluster_tree,max_depth = cluster.get_cluster_tree()
        if print_tree:
            print("Printing the tree")
            cluster.show_clutser_tree()
        return cluster,cluster_tree,max_depth

    def build_BIRCH(self):
        goal_name = utils.get_goal(self.goal, self.config)
        # self.attr_df = self.attr_df.drop(goal_name, axis = 1)
        # print(goal_name,self.attr_df.columns)
        cluster,cluster_tree,max_depth = self.cluster_driver(self.attr_df)
        return cluster,cluster_tree,max_depth

    
    def bellwether(self,selected_projects,all_projects):
        final_score = {}
        final_model = {}
        final_features = {}
        count = 0
        for s_project in selected_projects:
            try:
                data = self.prepare_data(s_project)
                data, cols = self.apply_cfs(data)
                list_temp, model_touse,sa = DECART_bellwether_CFS(data, self.metrics, 
                                            self.month, all_projects, s_project, 
                                            self.directory, self.goal, cols, self.config)
                final_score[s_project] = list_temp
                final_model[s_project] = model_touse
                final_features[s_project] = cols
            except ArithmeticError as e:
                print(e)
                continue
        return [final_score, final_model, final_features]

    def run_bellwether(self,projects):
        threads = []
        results = {}
        models = {}
        features = {}
        _projects = projects
        split_projects = np.array_split(_projects, self.cores)
        # split_projects = [arr for arr in split_projects if len(arr) > 0]
        for i in range(self.cores):
            print("starting thread ",i)
            t = ThreadWithReturnValue(target = self.bellwether, args = [split_projects[i],projects])
            threads.append(t)
        for th in threads:
            th.start()
        for th in threads:
            response = th.join()
            print(response)
            results.update(response[0])
            models.update(response[1])
            features.update(response[2])
        return results, models, features

    # For every cluster : Train model on a project and test on other projects
    # store in .pkl file
    def run(self,selected_projects,cluster_id,data_store_path):
        print(cluster_id)
        final_score, models, features = self.run_bellwether(selected_projects)
        data_path = Path(data_store_path + utils.get_goal(self.goal, self.config) + '/' + str(cluster_id))
        if not data_path.is_dir():
            os.makedirs(data_path)
        with open(data_store_path + utils.get_goal(self.goal, self.config) + '/' + str(cluster_id) + '/goal_' + str(self.goal)  + '.pkl', 'wb') as handle:
            pickle.dump(final_score, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(data_store_path + utils.get_goal(self.goal, self.config) + '/' + str(cluster_id) + '/goal_' + str(self.goal)  + '_models.pkl', 'wb') as handle:
            pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(data_store_path + utils.get_goal(self.goal, self.config) + '/' + str(cluster_id) + '/goal_' + str(self.goal)  + '_features.pkl', 'wb') as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # df = pd.read_pickle(data_store_path + str(cluster_id)  + '/700_RF_default_bellwether.pkl')


def birch_bellwether_p_CFS(config=DEFAULT_CONFIG):
    month = config.month
    no_goals = config.no_goals
    seed = config.seed

    for i in range(no_goals):
        print('Running Goal:', i)
        goal = utils.get_goal(i, config)
        start = timeit.default_timer()
        path = "data/data_use/"
        meta_path = 'results/attribute/'+str(seed)+'/data_attribute_' + goal + '.csv'
        data_store_path = 'results/with_CFS_DE/'+str(seed)+'/month_' + str(month) + '_models/'
        attr_df = pd.read_csv(meta_path, index_col=0)

        attr_df_index = list(attr_df.index)
        sample_size = attr_df.shape[0]

        print("Training size: ", int(sample_size*0.8))

        training_projects = random.sample(attr_df_index, int(sample_size*0.8)) 
        test_projects = []
        test_len=0
        for project in attr_df_index:
            if project not in training_projects:
                test_projects.append(project) #todo use seed rand

        attr_df_train = attr_df.loc[training_projects]
        attr_df_test = attr_df.loc[test_projects]

        bell = Bellwether(path,attr_df_train,i,month,config)
        cluster,cluster_tree,max_depth = bell.build_BIRCH()

        print("Maximum depth: ", max_depth)
        print(cluster_tree)

        # Collect the leaf nodes ids in cluster_ids
        cluster_ids = []
        for key in cluster_tree:
            if cluster_tree[key].depth == max_depth:
                cluster_ids.append(key)

        # For each leaf node, run bellwether
        for ids in cluster_ids:
            selected_projects = list(attr_df_train.loc[cluster_tree[ids].data_points].index)
            print(selected_projects)
            bell.run(selected_projects,ids,data_store_path)

        data_path = Path(data_store_path + goal)
        if not data_path.is_dir():
            os.makedirs(data_path)

        attr_df_train.to_pickle(data_store_path + '/' + goal + '/train_data.pkl')
        attr_df_test.to_pickle(data_store_path + '/' + goal + '/test_data.pkl')

        stop = timeit.default_timer() 
        print("Model training time: ", stop - start)


if __name__ == "__main__":
    month = 12 #todo 12 and 24 months into the future
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
            print('Running Goal:', i)
            goal = utils.get_goal(i, DEFAULT_CONFIG)
            start = timeit.default_timer()
            path = 'data/data_use/'
            meta_path = 'results/attribute/'+str(seed)+'/data_attribute_' + goal + '.csv'
            data_store_path = 'results/with_CFS_DE/'+str(seed)+'/month_' + str(month) + '_models/'
            attr_df = pd.read_csv(meta_path, index_col=0)

            attr_df_index = list(attr_df.index)
            training_projects = random.sample(attr_df_index, 34) #124todo change to files number previously 1200
            test_projects = []
            test_len=0
            for project in attr_df_index:
                if project not in training_projects:
                    test_projects.append(project) #todo use seed rand

            attr_df_train = attr_df.loc[training_projects]
            attr_df_test = attr_df.loc[test_projects]

            bell = Bellwether(path,attr_df_train,i,month, DEFAULT_CONFIG)
            cluster,cluster_tree,max_depth = bell.build_BIRCH()

            print("Maximum depth: ", max_depth)
            print(cluster_tree)

            # Collect the leaf nodes ids in cluster_ids
            cluster_ids = []
            for key in cluster_tree:
                if cluster_tree[key].depth == max_depth:
                    cluster_ids.append(key)

            # For each leaf node, run bellwether
            for ids in cluster_ids:
                selected_projects = list(attr_df_train.loc[cluster_tree[ids].data_points].index)
                print(selected_projects)
                bell.run(selected_projects,ids,data_store_path)

            data_path = Path(data_store_path + goal)
            if not data_path.is_dir():
                os.makedirs(data_path)

            attr_df_train.to_pickle(data_store_path + '/' + goal + '/train_data.pkl')
            attr_df_test.to_pickle(data_store_path + '/' + goal + '/test_data.pkl')

            stop = timeit.default_timer() 
            print("Model training time: ", stop - start)
