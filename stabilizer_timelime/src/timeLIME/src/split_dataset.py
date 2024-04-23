import os
import numpy
import pandas as pd

# Input folder path containing CSV files
input_folder = 'H:\RAISE\Spring 24\TimeLIME_for_CS\data_paul'

# Output folder path to store split files
output_folder = 'H:\RAISE\Spring 24\TimeLIME_for_CS\data'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through each CSV file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Sort the DataFrame by the "Dates" column
        df.sort_values(by='dates', inplace=True)

        # Calculate the number of rows in the file
        n = len(df)

        # Split the file into three parts
        split_1 = df.iloc[:n - 13]
        split_2 = df.iloc[n - 13:n - 1]
        split_3 = df.iloc[n - 12:]

        filename = filename.split('.')[0]

        # Define the output file names
        split_filename_1 = f'{filename}_split_1.csv'
        split_filename_2 = f'{filename}_split_2.csv'
        split_filename_3 = f'{filename}_split_3.csv'

        # Create the output file paths
        split_path_1 = os.path.join(output_folder, split_filename_1)
        split_path_2 = os.path.join(output_folder, split_filename_2)
        split_path_3 = os.path.join(output_folder, split_filename_3)

        # Save the split DataFrames to CSV files without the index column
        split_1.to_csv(split_path_1, index=False)
        split_2.to_csv(split_path_2, index=False)
        split_3.to_csv(split_path_3, index=False)

        print(f'Split files created for {filename}:')
        print(f'- {split_filename_1} ({len(split_1)} rows)')
        print(f'- {split_filename_2} ({len(split_2)} rows)')
        print(f'- {split_filename_3} ({len(split_3)} rows)')
