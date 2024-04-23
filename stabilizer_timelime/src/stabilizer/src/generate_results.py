import os

# from .scott_knott import run
from .scott_knott import run
import csv


def generate_combined_files(seeds: list):
    source_folder = "results/with_CFS_DE/{}/Stats_new/"
    destination_folder = "results/with_CFS_DE/Stats_new/"

    for name in seeds:
        source_path = source_folder.format(name)

        for filename in os.listdir(source_path):
            if os.path.isfile(os.path.join(source_path, filename)):
                dest_file_path = os.path.join(destination_folder, filename)
                src_file_path = os.path.join(source_path, filename)

                with open(src_file_path, 'r') as src_file, open(dest_file_path, 'a') as dest_file:
                    dest_file.write(src_file.read())

    folder_path = "results/with_CFS_DE/Stats_new"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        merged_items = {}

        with open(file_path, 'r') as f:
            lines = f.readlines()
            i=0
            while i < len(lines):
                item_name = lines[i].strip()
                item_values = []
                i += 1
                while i < len(lines) and lines[i].strip():
                    item_values.append(lines[i].strip())
                    i += 1
                merged_items[item_name] = merged_items.get(item_name, []) + item_values
                i += 1

        with open(os.path.join(folder_path, "combined_" + filename), 'w') as f:
            for item, values in merged_items.items():
                f.write(item + "\n")
                f.write(" ".join(values) + "\n\n")


def generate_results(seeds: list, goals: list):
    # Merge files together
    generate_combined_files(seeds)

    metrics = ["sa", "mre"]

    for goal in goals:
        for metric in metrics:
            filepath = "results/with_CFS_DE/Stats_new/combined_{0}_{1}.txt".format(goal, metric)

            if not os.path.isfile(filepath):
                print(filepath, " is not a valid path/ does not exist")
                continue
            
            run(filename=filepath)
            res = run(filename=filepath)

            with open("{0}_{1}.csv".format(goal, metric), "w") as f:
                writer = csv.writer(f)
                writer.writerow(["rank", "treatment", "Median", "IQR", "10th Perc.", "25th Perc.", "75th Perc.", "90th Perc."])
                writer.writerows(res)