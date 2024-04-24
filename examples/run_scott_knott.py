from stabilizer_timelime.src.stabilizer import collect_bellwether
from stabilizer_timelime.src.stabilizer import generate_results

month = 12
goals = ['monthly_contributors', 'monthly_commits',  'monthly_open_PRs', 'monthly_closed_PRs', 'monthly_open_issues', 'monthly_closed_issues', 'monthly_stargazer']
seeds = [12345]

collect_bellwether(seeds=seeds, month=month, goals=goals)
generate_results(seeds=seeds, goals=goals)