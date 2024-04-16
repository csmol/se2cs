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

from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
import sys
import os
import timeit


import matplotlib.pyplot as plt

from . import birch
from .predictor_advance_v1 import *
from . import utils


from multiprocessing import  cpu_count
from threading import Thread

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
            self._return = self._target(*self._args,**self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return



class Bellwether(object):

    def __init__(self,data_path,attr_df, goal, month, config=DEFAULT_CONFIG):
        self.directory = data_path
        self.attr_df = attr_df
        self.cores = cpu_count()
        self.goal = goal
        self.metrics = 0
        self.month = month
        self.config = config

            
    def prepare_data(self, repo_name):
        df_raw = pd.read_csv(self.directory + repo_name, sep=',')
        df_raw = df_raw.drop(columns=['dates'])  
        last_col = utils.get_goal(self.goal, self.config)
        cols = list(df_raw.columns.values)
        cols.remove(last_col)
        df_adjust = df_raw[cols+[last_col]]
        return df_adjust


    # Cluster Driver
    def cluster_driver(self,df,print_tree = True):
        X = df.apply(pd.to_numeric)
        cluster = birch.birch(branching_factor=self.config.branching_factor)
        cluster.fit(X)
        cluster_tree,max_depth = cluster.get_cluster_tree()
        if print_tree:
            cluster.show_clutser_tree()
        return cluster,cluster_tree,max_depth

    def build_BIRCH(self):
        goal_name = utils.get_goal(self.goal, self.config)
        # self.attr_df = self.attr_df.drop(goal_name, axis = 1)
        # print(goal_name,self.attr_df.columns)
        cluster,cluster_tree,_ = self.cluster_driver(self.attr_df)
        return cluster,cluster_tree

    
    def bellwether(self,selected_projects,all_projects):
        final_score = {}
        final_model = {}
        count = 0
        for s_project in selected_projects:
            try:
                data = self.prepare_data(s_project)
                print(s_project)
                list_temp, model_touse, sa = DECART_bellwether(data, self.metrics,self.month, all_projects, s_project,self.directory, self.goal, self.config)
                final_score[s_project] = list_temp
                final_model[s_project] = model_touse
            except ArithmeticError as e:
                print(e)
                continue
        return [final_score, final_model]

    def run_bellwether(self,projects):
        threads = []
        results = {}
        models = {}
        _projects = projects
        split_projects = np.array_split(_projects, self.cores)
        for i in range(self.cores):
            print("starting thread ",i)
            t = ThreadWithReturnValue(target = self.bellwether, args = [split_projects[i],projects])
            threads.append(t)
        for th in threads:
            th.start()
        for th in threads:
            response = th.join()
            results.update(response[0])
            models.update(response[1])
        return results,models

    def run(self,selected_projects,cluster_id,data_store_path):
        print(cluster_id)
        final_score, models = self.run_bellwether(selected_projects)
        data_path = Path(data_store_path + utils.get_goal(self.goal, self.config) + '/' + str(cluster_id))
        if not data_path.is_dir():
            os.makedirs(data_path)
        with open(data_store_path + utils.get_goal(self.goal, self.config) + '/' + str(cluster_id) + '/goal_' + str(self.goal)  + '.pkl', 'wb') as handle:
            pickle.dump(final_score, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(data_store_path + utils.get_goal(self.goal, self.config) + '/' + str(cluster_id) + '/goal_' + str(self.goal)  + '_models.pkl', 'wb') as handle:
            pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # df = pd.read_pickle(data_store_path + str(cluster_id)  + '/700_RF_default_bellwether.pkl')


def default_bellwether(config=DEFAULT_CONFIG):
    month = config.month
    no_goals = config.no_goals
    seed = config.seed
    cores = cpu_count()


    for i in range(no_goals):
        print('Running Goal:', i)
        goal = utils.get_goal(i, config)
        start = timeit.default_timer()
        path = "data/data_use/"
        meta_path = 'results/with_CFS_DE/'+str(seed)+'/month_' + str(month) + '_models/' + goal + '/train_data.pkl'
        data_store_path = 'results/with_CFS_DE/'+str(seed)+'/month_' + str(month) + '_models/'
        attr_df = pd.read_pickle(meta_path)
        project_list = list(attr_df.index)
        project_list = project_list

        threads = []
        results = {}
        models = {}
        split_projects = np.array_split(project_list, cores)

        bell = Bellwether(path,attr_df,i,month, config)
        for i in range(cores):
            print("starting thread ",i)
            t = ThreadWithReturnValue(target = bell.bellwether, args = [split_projects[i],project_list])
            threads.append(t)
        for th in threads:
            th.start()
        for th in threads:
            response = th.join()
            results.update(response[0])
            models.update(response[1])
                    
        print("default path : ",data_store_path ," ", goal,'/default_bellwether.pkl')

        with open(data_store_path + goal + '/default_bellwether.pkl', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(data_store_path + goal + '/default_bellwether_models.pkl', 'wb') as handle:
            pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    month = 12
    no_goals= 7
    cores = cpu_count()
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
            meta_path = 'results/with_CFS_DE/'+str(seed)+'/month_' + str(month) + '_models/' + goal + '/train_data.pkl'
            data_store_path = 'results/with_CFS_DE/'+str(seed)+'/month_' + str(month) + '_models/'
            attr_df = pd.read_pickle(meta_path)
            project_list = list(attr_df.index)
            project_list = project_list

            threads = []
            results = {}
            models = {}
            split_projects = np.array_split(project_list, cores)

            bell = Bellwether(path,attr_df,i,month, DEFAULT_CONFIG)
            for i in range(cores):
                print("starting thread ",i)
                t = ThreadWithReturnValue(target = bell.bellwether, args = [split_projects[i],project_list])
                threads.append(t)
            for th in threads:
                th.start()
            for th in threads:
                response = th.join()
                results.update(response[0])
                models.update(response[1])
            
            print(results)
            
            print("default path : ",data_store_path ," ", goal,'/default_bellwether.pkl')

            with open(data_store_path + goal + '/default_bellwether.pkl', 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            with open(data_store_path + goal + '/default_bellwether_models.pkl', 'wb') as handle:
                pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)






