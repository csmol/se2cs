import pandas as pd
import numpy as np

from . import utils

import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
import sys
import os
from ..config import Config

DEFAULT_CONFIG = Config()

def calculate_median(repo_pool, path, goal, config):
    feature_selected = []
    repos = []
    for repo in sorted(repo_pool):
        try:
            data = utils.data_goal_arrange(repo, path, goal, config)
            data = data.iloc[:,:-1]
            data_median = data.median().values.tolist()
            feature_selected.append(data_median)
            repos.append(repo)
        except Exception as e:
            print(e)
            continue
    return feature_selected, repos

def attribute_selector(config=DEFAULT_CONFIG):
    repo_pool = []
    path = "data/data_use/"
    no_goals= config.no_goals
    seed = config.seed

    for filename in os.listdir(path):
        if not filename.startswith('.'):
            repo_pool.append(os.path.join(filename))

    for _goal in range(no_goals):
        goal = utils.get_goal(_goal, config)
        print("goal: ",goal)
        df_cols = utils.data_goal_arrange(repo_pool[0], path, _goal, config).columns[:-1]
        feature_selected, repos = calculate_median(repo_pool, path, _goal, config)
        print(feature_selected, repos)
        median_df = pd.DataFrame(feature_selected, columns = df_cols, index = repos)
        median_df.to_csv('results/attribute/'+str(seed)+'/data_attribute_' + goal + '.csv')


if __name__ == '__main__':
    repo_pool = []
   
    path = 'data/data_use/'
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
        for filename in os.listdir(path):
            if not filename.startswith('.'):
                repo_pool.append(os.path.join(filename))

        for _goal in range(no_goals):
            goal = utils.get_goal(_goal, DEFAULT_CONFIG)
            print("goal: ",goal)
            df_cols = utils.data_goal_arrange(repo_pool[0], path, _goal, DEFAULT_CONFIG).columns[:-1]
            print("df_cols: ",len(df_cols))
            feature_selected, repos = calculate_median(repo_pool, path, _goal, DEFAULT_CONFIG)
            median_df = pd.DataFrame(feature_selected, columns = df_cols, index = repos)
            median_df.to_csv('results/attribute/'+str(seed)+'/data_attribute_' + goal + '.csv')
