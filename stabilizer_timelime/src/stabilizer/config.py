DEFAULT_GOALS = ['monthly_contributors', 'monthly_commits',  'monthly_open_PRs', 'monthly_closed_PRs', 'monthly_open_issues', 'monthly_closed_issues', 'monthly_stargazer']

class Config:
    def __init__(self, goals:list=DEFAULT_GOALS, month:int=12, data_path:str="data/data_use/", branching_factor:int=7, data_meta:str=None, domain:list=None, seed=12345) -> None:
        self.goals = goals
        self.no_goals = len(goals)
        self.month = month
        self.data_path = data_path
        self.branching_factor = branching_factor
        self.seed = seed
        
    def __str__(self) -> str:
        res = "Configuration:\n"
        res += "Goals: " + " ,".join(self.goals) + "\n"
        res += "Number of Goals: " + str(self.no_goals) + "\n"
        res += "Branching Factor: " + str(self.branching_factor) + "\n"
        res += "Seed: " + str(self.seed) + "\n"
        res += "Data Path: " + str(self.data_path) + "\n"
        res = "="*50 + "\n" + res + "="*50 + "\n"

        return res 
