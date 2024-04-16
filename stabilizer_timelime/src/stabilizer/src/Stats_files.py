import pandas as pd
import numpy as np
import math
import pickle
import sys
from . import utils


from ..config import Config

DEFAULT_CONFIG = Config()

def create_files(path, goal,seed, month):
    for criteria in ['mre','sa']:
        with open(path+ str(seed) + '/Stats_new/' + goal + '_' + criteria + '.txt', 'w') as f:
            results_df = pd.read_csv(path +str(seed)+'/month_' + str(month) + '_models/' + goal + '/results_tuned_final_sa.csv')
            results_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
            for level in results_df.level.unique():
                sub_df = results_df[results_df['level'] == level]
                for col in ['self', 'bellwether','conv_bell','global']:
                    values = sub_df[col + '_' + criteria].values.tolist()
                    key = 'month_' + str(month) + '_level_' + str(level) + '_' + col
                    if col == 'self' + '_' + criteria:
                        key = col + '_level_' + str(level)
                    f.write("%s \n" % key)
                    for i in values:
                        f.write("%s " % i)
                    f.write("\n\n")   
                    if col == 'self':
                        seen_self = True


def generate_stats_files(config=DEFAULT_CONFIG):
    no_goals = config.no_goals
    seed = config.seed
    path = 'results/with_CFS_DE/'
    for i in range(no_goals):
        goal = utils.get_goal(i, config)
        create_files(path, goal,seed,config.month)
    


if __name__ == "__main__":
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
        path = 'results/with_CFS_DE/'
        for i in range(no_goals):
            goal = utils.get_goal(i)
            print("goal:", goal)
            create_files(path, goal,seed,12)