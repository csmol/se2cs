from stabilizer_timelime.src.dataslurper import create_dataset

csv_file = "all_repos_0315.csv"
access_tokens = {}

create_dataset(repo_list_csv=csv_file, access_tokens_dict=access_tokens, dest_folder="data_paul/")
