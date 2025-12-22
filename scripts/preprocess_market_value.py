import os
import pandas as pd

# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the raw data
raw_dir = os.path.join(script_dir, '..', 'data', 'raw', 'player_market_value', 'player_market_value.csv')

# Path to the processed data directory
processed_dir = os.path.join(script_dir, '..', 'data', 'processed', 'player_market_value_clean.csv')

# Load CSV
df = pd.read_csv(raw_dir)

# Basic inspection
print(df.info())
print(df.isna().sum())

#Fix data types
df['date_unix'] = pd.to_datetime(df['date_unix'], errors='coerce')
df['player_id'] = df['player_id'].astype(int)

# Drop rows with missing important values
df = df.dropna(subset=['player_id', 'date_unix', 'value'])

# Sort rows by player_id first (group each player together),
# then by date_unix so each player's records are in chronological order
df = df.sort_values(by=['player_id', 'date_unix'])

# Save cleaned version
df.to_csv(processed_dir, index=False)

print("Saved clean market value data")