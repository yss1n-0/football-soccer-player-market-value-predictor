import os
import pandas as pd
from datetime import datetime

def season_to_year(season):
    current_year = datetime.now().year
    if '/' in season:
        start = int(season.split('/')[0])
        return 2000 + start if (2000 + start) <= current_year + 1 else 1900 + start  # adjust for 20xx vs 19xx
    else:
        return int(season)  # if season is already a full year

# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the raw data directory
raw_dir = os.path.join(script_dir, '..', 'data', 'raw', 'player_performances', 'player_performances.csv')

# Path to the processed data directory
processed_dir = os.path.join(script_dir, '..', 'data', 'processed', 'player_performances_clean_2000.csv')

# Chunk size
chunksize = 100_000
first_chunk = True # For writing header only once

# columns = ['player_id', 'season_name', 'competition_id', 'competition_name',
#            'team_id', 'team_name', 'nb_in_group', 'nb_on_pitch', 'goals',
#            'assists', 'own_goals', 'subed_in', 'subed_out', 'yellow_cards',
#            'second_yellow_cards', 'direct_red_cards', 'penalty_goals',
#            'minutes_played', 'goals_conceded', 'clean_sheets']

# Process the CSV file in chunks
for chunk in pd.read_csv(raw_dir, chunksize=chunksize):
    # Drop rows with missing critical values
    chunk = chunk.dropna(subset=['player_id', 'season_name', 'team_id'])

    # Convert season to start year
    chunk['season_start_year'] = chunk['season_name'].apply(season_to_year)

    # Keep only seasons starting in 2000 or later
    chunk = chunk[chunk['season_start_year'] >= 2000]

    # Convert 'season_start_year' to int
    chunk['season_start_year'] = chunk['season_start_year'].astype(int)

    # Sort by player_id and season_start_year
    chunk = chunk.sort_values(by=['player_id', 'season_start_year'])

    # Drop rows with invalid or missing season years
    chunk = chunk.dropna(subset=['season_start_year'])

    # Drop the helper column
    chunk = chunk.drop(columns=['season_start_year'])

    # For the first chunk only
    if first_chunk:
        # Basic inspection
        print(chunk.info())
        print(chunk.isna().sum())

        # Save the first chunk to CSV including the header (column names)
        chunk.to_csv(processed_dir, index=False)

        # Set first_chunk to False to ensure subsequent chunks do not print headers or basic inspection details
        first_chunk = False
        
    # For all remaining chunks:
    else:
        # Append to the same CSV file
        chunk.to_csv(processed_dir, index=False)

print("Saved player performances data")