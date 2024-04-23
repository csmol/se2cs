from stabilizer_timelime.src.stabilizer import config
from stabilizer_timelime.src.stabilizer import init_datafiles, attribute_selector, birch_bellwether_p_CFS
from stabilizer_timelime.src.stabilizer import default_bellwether, find_bellwether, collect_bellwether
from stabilizer_timelime.src.stabilizer import performance_calculator, generate_stats_files, generate_results

config = config.Config(data_path="data_paul/", branching_factor=20)
print(config)
init_datafiles(config=config)
attribute_selector(config=config)
birch_bellwether_p_CFS(config=config)
default_bellwether(config=config)
find_bellwether(config)
collect_bellwether(config)
performance_calculator(config)
generate_stats_files(config)
