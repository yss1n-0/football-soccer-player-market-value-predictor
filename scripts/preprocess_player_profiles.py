import os
import pandas as pd

# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the raw data
raw_dir = os.path.join(script_dir, '..', 'data', 'raw', 'player_profiles', 'player_profiles.csv')

# Path to the processed data directory
processed_dir = os.path.join(script_dir, '..', 'data', 'processed', 'player_profiles_clean.csv')

# Load CSV
df = pd.read_csv(raw_dir)

# Basic inspection
print(df.info())
print(df.isna().sum())

# Fix data types
date_cols = [
    'date_of_birth',
    'joined',
    'contract_expires',
    'date_of_death',
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Drop rows with missing important values
df = df.dropna(subset=['player_id', 'player_name', 'position', 'main_position'])

# Drop extremely sparse columns
cols_to_drop = [
    'outfitter',
    'social_media_url',
    'player_agent_id',
    'player_agent_name',
    'contract_option',
    'date_of_last_contract_extension',
    'on_loan_from_club_id',
    'on_loan_from_club_name',
    'contract_there_expires',
    'second_club_url',
    'second_club_name',
    'third_club_url',
    'third_club_name',
    'fourth_club_url',
    'fourth_club_name',
]

for col in cols_to_drop:
    if col in df.columns:
        df = df.drop(columns=col)

# Save cleaned version
df.to_csv(processed_dir, index=False)

print("Saved clean player profiles data")