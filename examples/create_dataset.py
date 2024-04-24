from stabilizer_timelime.src.dataslurper import create_dataset
from stabilizer_timelime.src.stabilizer import init_datafiles


month = 12
goals = ['monthly_contributors', 'monthly_commits',  'monthly_open_PRs', 'monthly_closed_PRs', 'monthly_open_issues', 'monthly_closed_issues', 'monthly_stargazer']
seeds = [12345]
data_path = "data_paul/"


csv_file = "all_repos_0315.csv"
access_tokens = {}

create_dataset(repo_list_csv=csv_file, access_tokens_dict=access_tokens, dest_folder="data_paul/")

init_datafiles(data_path=data_path, seeds=seeds, month=month, goals=goals)