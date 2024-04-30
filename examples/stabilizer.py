from stabilizer_timelime.src.stabilizer import config
from stabilizer_timelime.src.stabilizer import attribute_selector, birch_bellwether_p_CFS
from stabilizer_timelime.src.stabilizer import default_bellwether, find_bellwether
from stabilizer_timelime.src.stabilizer import performance_calculator, generate_stats_files

# Knobs : month, goals, branching factor for birch clustering, random seed, source data path
config = config.Config(data_path="data_paul/", branching_factor=20, seed=12345)
print(config)

# Calculates the median for every goal across the project
attribute_selector(config=config)

# Performs the clustering and saves in the file
birch_bellwether_p_CFS(config=config)

# Calculates default bellwether/exemplar for every cluster
default_bellwether(config=config)

# Calculates level 0, level 1 bellwether
find_bellwether(config)

# Calculates MREs and SAs for every goal
performance_calculator(config)

# Collate the stats files
generate_stats_files(config)
