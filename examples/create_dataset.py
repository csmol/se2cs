from stabilizer_timelime.src.dataslurper import create_dataset
from stabilizer_timelime.src.stabilizer import init_datafiles

# Knobs to change : Make sure these parameters are consistent across all the files
month = 12
goals = ['monthly_contributors', 'monthly_commits',  'monthly_open_PRs', 'monthly_closed_PRs', 'monthly_open_issues', 'monthly_closed_issues', 'monthly_stargazer']
seeds = [12345]
data_path = "data_paul/"

# The csv file that has list of all the projects. 3 must columns - Organization, Project, and Link
csv_file = "all_repos_0315.csv"

# Github access tokens in the form of "username": "TOKEN" key-value pair
access_tokens = {}

# This function reads the list of projects, performs prudence check, and slurps the monthly data
# Currently it uses default prudence check parameters. To alter please refer to PrudenceChecker class at stabilizer_timelime/src/dataslurper/PrudenceChecker.py
create_dataset(repo_list_csv=csv_file, access_tokens_dict=access_tokens, dest_folder="data_paul/")

# This function generates the necessary folder structure for the stabilizer
init_datafiles(data_path=data_path, seeds=seeds, month=month, goals=goals)