import os
import pandas as pd

# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the raw data directory
raw_dir = os.path.join(script_dir, '..', 'data', 'raw')

# List of CSV files to load
csv_files = ['player_market_value.csv',
             'player_performances.csv',
             'player_profiles.csv',
]

# Dictionary for DataFrames (key = file name, value = DataFrame)
dataframes = {}

# Loop through each CSV filedata
for file_name in csv_files:
    # Folder name matches the CSV name
    folder = file_name.replace(".csv", "")

    # Build the full path to the CSV file
    file_path = os.path.join(raw_dir, folder, file_name)

    if file_name == 'player_profiles.csv':
        dataframes[file_name] = pd.read_csv(
            file_path,
            dtype={
                'third_club_url': str,
                'third_club_name': str,
                'fourth_club_url': str,
                'fourth_club_name': str,
            }
        )
    else:
        # Read the CSV and store it in the dictionary
        dataframes[file_name] = pd.read_csv(file_path)


# Print first five rows of each DataFrame to check the data
for name, df in dataframes.items():
    print(f"\n--- {name} ---")
    print(df.head())

for name, df, in dataframes.items():
    print(f"\n--- {name} ---")

    # Print numbr of rows
    print("Rows:", df.shape[0])

    # Column names
    print("Columns:")
    print(list(df.columns))
