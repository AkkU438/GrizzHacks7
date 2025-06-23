import pandas as pd
import os

# Set the directory where your CSV files are
folder_path = 'Data/Drafted'

skip_file = '2024_Drafted_RB.csv'

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# List to hold each CSV as a DataFrame
dataframes = []

# Read and store each CSV into the list
for file in csv_files:
    if file == skip_file:
        print(f"Skipping file: {file}")
        continue  # Skip the specified file
    df = pd.read_csv(os.path.join(folder_path, file))
    dataframes.append(df)

# Combine all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a new CSV
combined_df.to_csv('Data/Drafted/all_drafted_RB.csv', index=False)
