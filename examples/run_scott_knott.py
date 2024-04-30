from stabilizer_timelime.src.stabilizer import collect_bellwether
from stabilizer_timelime.src.stabilizer import generate_results

# Knobs: Make sure these parameters are consistent across all the files
month = 12
goals = ['monthly_contributors', 'monthly_commits',  'monthly_open_PRs', 'monthly_closed_PRs', 'monthly_open_issues', 'monthly_closed_issues', 'monthly_stargazer']
seeds = [12345]

# Collects the level 0 and level 1 bellwether and stores in bellwether folders/
collect_bellwether(seeds=seeds, month=month, goals=goals)

# Generates the final mre and SA and ranks the treatment by applying scott-knott test
generate_results(seeds=seeds, goals=goals)