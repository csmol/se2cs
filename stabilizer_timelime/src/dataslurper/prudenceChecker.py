import requests
from urllib.parse import urlparse, parse_qs
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

class PrudenceChecker():
    def __init__(self, devs=2, prs=5, issues=10, releases=1, commits=20, duration=12) -> None:
        self.devs = devs
        self.prs = prs
        self.issues = issues
        self.releases = releases
        self.commits = commits
        self.duration = duration
    
    def get_response(self, url, auth=None, headers=None, state=None):
        url = "{url}?&per_page=30&page=1".format(url=url)
        if state is not None:
            url = url + "&state={state}".format(state=state)
        
        time.sleep(2)

        if auth is not None:
            response = requests.get(url, auth=auth)
        else:
            response = requests.get(url, headers=headers)
        res = response.json()

        while "next" in response.links.keys():
            time.sleep(2)
            url = response.links["next"]["url"]
            if auth is not None:
                response = requests.get(url, auth=auth)
            else:
                response = requests.get(url, headers=headers)
            res.extend(response.json())
        return res, len(res)
    

    def get_commits_count(self, url, auth=None, headers=None):
        url = "{url}?&per_page=30&page=1".format(url=url)
        response = requests.get(url, auth=auth)
        res = response.json()

        latest_commit = datetime.strptime(res[0]["commit"]["author"]["date"], "%Y-%m-%dT%H:%M:%SZ")

        last_page = 1

        if "last" in response.links.keys():
            url = response.links["last"]["url"]
            parsed_url = urlparse(url=url)
            last_page = int(parse_qs(parsed_url.query)["page"][0])
            response = requests.get(url, auth=auth)
        
        res = response.json()
        total_commit_count = (last_page-1)*30 + len(res)
        first_commit =  datetime.strptime(res[-1]["commit"]["author"]["date"], "%Y-%m-%dT%H:%M:%SZ")

        diff = relativedelta(latest_commit, first_commit)
        project_period_years = diff.years
        project_period_months = project_period_years*12 + diff.months

        return total_commit_count, project_period_years, project_period_months


    def get_count(self, url, auth=None, headers=None, state=None):
        if state is not None:
            url = "{url}?&per_page=30&page=1&state={state}".format(url=url, state=state)
        else:
            url = "{url}?&per_page=30&page=1".format(url=url)

        response = requests.get(url, auth=auth)
        res = response.json()
        
        last_page = 1

        if "last" in response.links.keys():
            url = response.links["last"]["url"]
            parsed_url = urlparse(url=url)
            last_page = int(parse_qs(parsed_url.query)["page"][0])
            response = requests.get(url, auth=auth)
        
        res = response.json()
        total_count = (last_page-1)*30 + len(res)
        
        return total_count


    def is_passing_prudence(self, url: str, token: str, username: str) -> bool:
        if not "github.com" in url:
            print("URL {0} is not a github URL".format(url))
            return False

        api_url = url.replace("https://github.com/","https://api.github.com/repos/")

        try:
            # no of contributors
            _, total_devs = self.get_response("{api_url}/contributors".format(api_url=api_url), auth=(username, token))
            
            # total commits
            total_commits, project_period_years, project_period_months  = self.get_commits_count("{api_url}/commits".format(api_url=api_url), auth=(username, token))

            # issues
            total_open_issues = self.get_count("{api_url}/issues".format(api_url=api_url), auth=(username, token), state="open")
            total_closed_issues = self.get_count("{api_url}/issues".format(api_url=api_url), auth=(username, token), state="closed")

            # PRs
            total_open_prs = self.get_count("{api_url}/pulls".format(api_url=api_url), auth=(username, token), state="open")
            total_closed_prs = self.get_count("{api_url}/pulls".format(api_url=api_url), auth=(username, token), state="closed")

            # releases
            total_releases = self.get_count("{api_url}/releases".format(api_url=api_url), auth=(username, token))

            if total_devs>=self.devs and (total_open_prs+total_closed_prs)>=self.prs and (total_open_issues+total_closed_issues)>self.issues and total_releases>self.releases and total_commits>self.commits and project_period_months>self.duration:
                return True
        
            return False

        except Exception as e:
            print("URL: {0} Exception: {1}".format(url, e.message()))
