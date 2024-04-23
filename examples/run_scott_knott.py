from stabilizer_timelime.src.stabilizer import config
from stabilizer_timelime.src.stabilizer import generate_results


config = config.Config(data_path="data_paul/", branching_factor=20)
generate_results(seeds=[12345], goals=config.goals)