import os
import pandas as pd

from . import utils

from ..config import Config

DEFAULT_CONFIG = Config()

def collect_bellwether(config=DEFAULT_CONFIG):
    month = config.month
    no_goals= config.no_goals
    seed = config.seed

    output_folder = "bellwethers/"

    for i in range(no_goals):
        goal = utils.get_goal(i, config)

        res_df_0 = pd.DataFrame()
        res_df_1 = pd.DataFrame()

        result_source = 'results/with_CFS_DE/'+str(seed)+'/month_' + str(month) + '_models/' + goal + '/'
        bellwether_0 = pd.read_csv(os.path.join(result_source, "bellwether_level_0.csv"))
        bellwether_1 = pd.read_csv(os.path.join(result_source, "bellwether_level_1.csv"))

        bellwether_0["seed"] = seed
        bellwether_1["seed"] = seed

        res_df_0 = res_df_0.append(bellwether_0, ignore_index=True)
        res_df_1 = res_df_1.append(bellwether_1, ignore_index=True)
        
        if not os.path.exists("meta_data/old_new_names_mapping.csv"):
            print("Old names - New names mapping missing. Exiting ...")
            exit(0)

        mapping_df = pd.read_csv("meta_data/old_new_names_mapping.csv")

        res_df_0 = pd.merge(res_df_0, mapping_df, how='left', left_on='bellwether', right_on='new')
        res_df_1 = pd.merge(res_df_1, mapping_df, how='left', left_on='bellwether', right_on='new')

        res_df_0.drop(columns=['Unnamed: 0', "id", "new"], inplace=True)
        res_df_0.rename(columns={"old":"bellwether_project"}, inplace=True)

        res_df_1.drop(columns=['Unnamed: 0', "id", "new"], inplace=True)
        res_df_1.rename(columns={"old":"bellwether_project"}, inplace=True)

        res_df_0.sort_values(by="bellwether", inplace=True)
        res_df_1.sort_values(by="bellwether", inplace=True)

        res_df_0.to_csv(os.path.join(output_folder,goal+"_level_0.csv"), index=False)
        res_df_1.to_csv(os.path.join(output_folder,goal+"_level_1.csv"), index=False)


if __name__ == "__main__":
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
    }

    month = 12
    no_goals= 7

    output_folder = "bellwethers/"

    for i in range(no_goals):
        goal = utils.get_goal(i, config=DEFAULT_CONFIG)

        res_df_0 = pd.DataFrame()
        res_df_1 = pd.DataFrame()

        for _, seed in seeds.items():
            seed = seed[0]
            result_source = 'results/with_CFS_DE/'+str(seed)+'/month_' + str(month) + '_models/' + goal + '/'
            bellwether_0 = pd.read_csv(os.path.join(result_source, "bellwether_level_0.csv"))
            bellwether_1 = pd.read_csv(os.path.join(result_source, "bellwether_level_1.csv"))

            bellwether_0["seed"] = seed
            bellwether_1["seed"] = seed

            res_df_0 = res_df_0.append(bellwether_0, ignore_index=True)
            res_df_1 = res_df_1.append(bellwether_1, ignore_index=True)
        
        mapping_df = pd.read_csv("old_new_names_mapping.csv")

        res_df_0 = pd.merge(res_df_0, mapping_df, how='left', left_on='bellwether', right_on='new')
        res_df_1 = pd.merge(res_df_1, mapping_df, how='left', left_on='bellwether', right_on='new')

        res_df_0.drop(columns=['Unnamed: 0', "id", "new"], inplace=True)
        res_df_0.rename(columns={"old":"bellwether_project"}, inplace=True)

        res_df_1.drop(columns=['Unnamed: 0', "id", "new"], inplace=True)
        res_df_1.rename(columns={"old":"bellwether_project"}, inplace=True)

        res_df_0.sort_values(by="bellwether", inplace=True)
        res_df_1.sort_values(by="bellwether", inplace=True)

        res_df_0.to_csv(os.path.join(output_folder,goal+"_level_0.csv"), index=False)
        res_df_1.to_csv(os.path.join(output_folder,goal+"_level_1.csv"), index=False)






