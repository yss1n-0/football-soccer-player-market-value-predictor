import os
import pandas as pd

# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the processed data directory
processed_dir = os.path.join(script_dir, '..', 'data', 'processed')

# Output path for the master dataset
output_path = os.path.join(processed_dir, 'master_dataset.csv')


# List of clean CSV files to load
clean_csvs = ['player_market_value_clean.csv',
             'player_performances_clean_2000.csv',
             'player_profiles_clean.csv',
]

# Dictionary for DataFrames (key = file name, value = DataFrame)
dataframes = {}
for file_name in clean_csvs:
    # Build the full path to the CSV file
    file_path = os.path.join(processed_dir, file_name)

    # Read the CSV and store it in the dictionary
    dataframes[file_name] = pd.read_csv(file_path)

# Assign DataFrames
df_market = dataframes['player_market_value_clean.csv']
df_perf = dataframes['player_performances_clean_2000.csv']
df_profiles = dataframes['player_profiles_clean.csv']

# Ensure date_unix is datetime so we can extract year
df_market['date_unix'] = pd.to_datetime(df_market['date_unix'], errors='coerce')

# Extract season start year (used as merge key)
df_market['season_start_year'] = df_market['date_unix'].dt.year

# Merge performances with market values
df_master = pd.merge(
    df_perf,
    df_market[['player_id','season_start_year','value']], 
    on=['player_id','season_start_year'],
    how='left'
    )

# Merge player profiles
df_master = pd.merge(
    df_master,
    df_profiles,
    on='player_id',
    how='left'
    )

# Create a binary column for loan, 1 if player is on loan this season, otherwise 0
df_master['on_loan'] = df_master['on_loan_from_club_id'].notna().astype(int)

# Fill club name for clarity
df_master['on_loan_from_club_name'] = df_master['on_loan_from_club_name'].fillna('None')

# Sort
df_master = df_master.sort_values(by=['player_id', 'season_start_year'])

# Fill logical NaNs
df_master['goals'] = df_master['goals'].fillna(0)
df_master['minutes_played'] = df_master['minutes_played'].fillna(0)

# Drop rows without market value (target variable)
df_master = df_master.dropna(subset=['value'])

# Quick sanity check
print(df_master.isna().sum())

# Save master dataset
df_master.to_csv(output_path, index=False)
print("Saved master dataset to data/processed/master_dataset.csv")