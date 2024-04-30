import shutil
from pathlib import Path

import os
import csv
import sys

import pandas as pd

def create_folders(seeds: list, month: int, goals: list):
    folders_to_create = ["data/data_use/","results/attribute/", "results/with_CFS_DE/", "meta_data/", "bellwethers/"]

    for folder in folders_to_create:
        shutil.rmtree(folder, ignore_errors=True)
        Path(folder).mkdir(parents=True, exist_ok=True)
    
    shutil.rmtree("data/data_use_{0}_months/".format(month), ignore_errors=True)
    Path("data/data_use_{0}_months/".format(month)).mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        attr_seed = os.path.join("results/attribute/", str(seed))
        Path(attr_seed).mkdir(parents=True, exist_ok=True)

        cfs_seed = os.path.join("results/with_CFS_DE/", str(seed))
        Path(cfs_seed).mkdir(parents=True, exist_ok=True)

    subfolders = ['Stats_new', 'month_{0}_models'.format(month)]

    for subfolder in subfolders:
        subfolder_path = os.path.join(cfs_seed, subfolder)
        if subfolder!='Stats_new':
            for goal in goals:
                updated_goal_path=os.path.join(subfolder_path, goal)
                os.makedirs(updated_goal_path, exist_ok=True)
        else:
            os.makedirs(subfolder_path, exist_ok=True)

    Path("results/with_CFS_DE/Stats_new").mkdir(parents=True, exist_ok=True)


def copy_source_2_data_use(data_path: str):
    source_dir = data_path
    dest_dir = "data/data_use/"

    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)
        if file_name.endswith('.csv'):
            shutil.copy(file_path, os.path.join(dest_dir, file_name))


def rename_data_files():
    directory = "data/data_use/"
    files = os.listdir(directory)

    files.sort()

    num = 0

    record = []

    # Loop through the files and rename them
    for file in files:
        # Create the new file name with leading zeros
        new_name = '{:03d}{}'.format(num, os.path.splitext(file)[1])
        # Rename the file
        os.rename(os.path.join(directory, file), os.path.join(directory, new_name))
        # Increment the number for the next file
        num += 1
        di = {"old":os.path.splitext(file)[0], "new":new_name}
        record.append(di)
    
    pd.DataFrame.from_dict(record).to_csv("meta_data/old_new_names_mapping.csv", index=False)


def move_x_months_data(month: int):
    source_folder = "data/data_use"
    destination_folder = "data/data_use_{0}_months/".format(month)

    for filename in os.listdir(source_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(source_folder, filename)
            df = pd.read_csv(filepath)

            num_rows = df.shape[0]

            if num_rows>month:
                shutil.copy(filepath, os.path.join(destination_folder, filename))

def move_back_2_data_use(month: int, goals: list):
    source_dir = "data/data_use_{0}_months/".format(month)
    dest_dir = "data/data_use/"

    for file_name in os.listdir(dest_dir):
        file_path = os.path.join(dest_dir, file_name)

        os.remove(file_path)

    cols = ["dates"] + goals

    # Loop through all files in the source directory
    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)
        
        # Check if the file is a CSV file
        if file_name.endswith('.csv'):

            df = pd.read_csv(file_path)
            df = df[cols]
            df = df.reindex(sorted(df.columns), axis=1)

            df.to_csv(file_path, index=False)
            
            shutil.copy(file_path, os.path.join(dest_dir, file_name))


def init_datafiles(data_path:str, seeds: list, month: int, goals: list):
    # Create necessary folders
    create_folders(seeds, month, goals)

    # Copy data files to data_use/
    copy_source_2_data_use(data_path)

    # Rename data files
    rename_data_files()

    # Only take data for x months
    move_x_months_data(month)

    # Move data back to data_use/ with some cleaning
    move_back_2_data_use(month, goals)