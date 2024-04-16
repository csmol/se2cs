import os
from pathlib import Path

from datetime import timedelta
from dateutil.relativedelta import relativedelta

import github
from github import Github
from github import Auth

import pandas as pd

from tqdm import tqdm

def inc_value_in_dict(result: dict, date: str, column: str) -> None:
    """
        Helper function that increments the indicator value for given date.
        Parameters:
            result: Python dictionary containing indicator value for a date
            date: String representation of the date
            coulmn: Indicator
        Returns:
            None 
    """

    if date not in result.keys():
        result[date] = {
            "monthly_contributors": 0,
            "monthly_commits": 0,
            "monthly_open_PRs": 0,
            "monthly_closed_PRs": 0,
            "monthly_merged_PRs": 0,
            "monthly_open_issues": 0,
            "monthly_closed_issues": 0,
            "monthly_stargazer": 0
            }
    
    result[date][column] += 1


def get_commits(repo: github.Repository.Repository, result: dict, contributors_list: dict) -> None:
    """
        Processes commit data for a repo
        Parameters:
            repo: Github Repository whose commits to process
            result: Python dictionary containing indicator value for a date
            contributors_list: Python dictionary containing list of authors every month
        Returns:
            None
    """
    commits = repo.get_commits()

    for commit in commits:
        commit_date = commit.commit.author.date.replace(day=1)
        commit_author = commit.commit.author.name

        inc_value_in_dict(result, commit_date.strftime('%Y-%m-%d'), "monthly_commits")
        contributors_list[commit_date.strftime('%Y-%m-%d')].add(commit_author)


def get_prs(repo: github.Repository.Repository, result: dict, contributors_list: dict) -> None:
    """
        Processes PR data for a repo
        Parameters:
            repo: Github Repository whose PRs to process
            result: Python dictionary containing indicator value for a date
            contributors_list: Python dictionary containing list of authors every month
        Returns:
            None
    """
    prs = repo.get_pulls(state="all")
    for pr in prs:
        pr_open_date = pr.created_at
        inc_value_in_dict(result, pr_open_date.replace(day=1).strftime('%Y-%m-%d'), "monthly_open_PRs")
        
        pr_merged_date = pr.merged_at
        if pr_merged_date is not None:
            inc_value_in_dict(result, pr_merged_date.replace(day=1).strftime('%Y-%m-%d'), "monthly_merged_PRs")
        
        pr_closed_date = pr.closed_at
        if pr_closed_date is not None:
            inc_value_in_dict(result, pr_closed_date.replace(day=1).strftime('%Y-%m-%d'), "monthly_closed_PRs")
    

def get_issues(repo: github.Repository.Repository, result: dict, contributors_list: dict) -> None:
    """
        Processes issues data for a repo
        Parameters:
            repo: Github Repository whose issues to process
            result: Python dictionary containing indicator value for a date
            contributors_list: Python dictionary containing list of authors every month
        Returns:
            None
    """
    issues = repo.get_issues(state="all")
    for issue in issues:
        issue_open_date = issue.created_at
        inc_value_in_dict(result, issue_open_date.replace(day=1).strftime('%Y-%m-%d'), "monthly_open_issues")

        issue_close_date = issue.closed_at
        if issue_close_date is not None:
            inc_value_in_dict(result, issue_close_date.replace(day=1).strftime('%Y-%m-%d'), "monthly_closed_issues")


def get_stargazers(repo: github.Repository.Repository, result: dict, contributors_list: dict) -> None:
    """
        Processes stargazers data for a repo
        Parameters:
            repo: Github Repository whose issues to process
            result: Python dictionary containing indicator value for a date
            contributors_list: Python dictionary containing list of authors every month
        Returns:
            None
    """
    stargazers = repo.get_stargazers_with_dates()
    for star in stargazers:
        starred_at = star.starred_at.replace(day=1)
        inc_value_in_dict(result, starred_at.strftime('%Y-%m-%d'), "monthly_stargazer")


def create_dataset(repo_list_csv: str, access_tokens: list, dest_folder: str) -> None:
    """
        Creates the dataset and stores in csv files in dest_folder. 
        Parameters:
            repo_list_csv: Path of the csv file that contains organization and repo name info
            access_tokens: List of github access tokens
            dest_folder: Path of the destination folder
    """
    token_ind = 0

    repo_list = pd.read_csv(repo_list_csv)

    print("Number of projects to slurp:", repo_list.shape[0])

    Path(dest_folder).mkdir(parents=True, exist_ok=True)

    if ("organization" not in repo_list.columns) or ("repo" not in repo_list.columns):
        raise Exception("CSV missing column organization or/and repo")

    funcs = [get_commits, get_prs, get_issues, get_stargazers]

    for _, row in tqdm(repo_list.iterrows()):

        org_name = row["organization"]
        repo_name = row["repo"]

        if os.path.isfile(os.path.join(dest_folder, org_name+"_"+repo_name+".csv")):
            continue

        auth = Auth.Token(access_tokens[token_ind%len(access_tokens)])

        g = Github(auth=auth)

        repo = g.get_repo(org_name+"/"+repo_name)

        created_at = repo.created_at
        pushed_at = repo.pushed_at

        first_commit = repo.get_commits().reversed[0]
        first_commit_date = first_commit.commit.author.date

        start_date = min(created_at, first_commit_date)

        result = {}
        contributors_list = {}

        current_date = start_date.replace(day=1)
        end_date = pushed_at.replace(day=1) + timedelta(days=32)

        while current_date<end_date:
            result[current_date.strftime('%Y-%m-%d')] = {
                "monthly_contributors": 0,
                "monthly_commits": 0,
                "monthly_open_PRs": 0,
                "monthly_closed_PRs": 0,
                "monthly_merged_PRs": 0,
                "monthly_open_issues": 0,
                "monthly_closed_issues": 0,
                "monthly_stargazer": 0,
                }
            
            contributors_list[current_date.strftime('%Y-%m-%d')] = set()
            
            current_date += relativedelta(months=+1)

        # For each indicator type use different token
        try:
            for fn in funcs:
                token_ind += 1
                auth = Auth.Token(access_tokens[token_ind%len(access_tokens)])
                g = Github(auth=auth)
                repo = g.get_repo(org_name+"/"+repo_name)
                fn(repo, result, contributors_list)

            for k in result.keys():
                result[k]["monthly_contributors"] = 0 if k not in contributors_list.keys() else len(contributors_list[k])

            res_df = pd.DataFrame(result).T
            res_df.reset_index(inplace=True)

            res_df.rename(columns={"index":"dates"})

            res_df.to_csv(os.path.join(dest_folder, org_name+"_"+repo_name+".csv"), index=False)
        except Exception as e:
            print("Error Project Name: " + repo_name)
            print("Error: ", e, " In function ", str(fn))
        
        token_ind += 1