import os
import pandas as pd

# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to master_dataset.csv
file_path = os.path.join(script_dir, '..', 'data', 'processed', 'master_dataset.csv')

output_path = os.path.join(script_dir, '..', 'data', 'processed', 'model_ready_dataset.csv')

# Load dataset
df_master = pd.read_csv(file_path)

# Fill missing small categorical columns
df_master['foot'] = df_master['foot'].fillna('Unknown')
df_master['position'] = df_master['position'].fillna('Unknown')
df_master['main_position'] = df_master['main_position'].fillna('Unknown')

df_master['is_eu'] = df_master['is_eu'].fillna(False)

# Make the is_eu column's datatype is a boolean
df_master['is_eu'] = df_master['is_eu'].astype(bool)

# Convert contract_expires to datetime
df_master['contract_expires'] = pd.to_datetime(df_master['contract_expires'], errors='coerce')

# Create a new column: years remaining on contract at the time of the season
df_master['contract_remaining_years'] = (
    df_master['contract_expires'].dt.year - df_master['season_start_year']
)

# Remove negative values (expired contracts before the season)
df_master['contract_remaining_years'] = df_master['contract_remaining_years'].clip(lower=0)

# Convert date_unix to datetime
df_master['date_unix'] = pd.to_datetime(df_master['date_unix'], errors='coerce')

# Handle date columns for features like player age in a season
df_master['date_of_birth'] = pd.to_datetime(df_master['date_of_birth'], errors='coerce')
df_master['age'] = (df_master['date_unix'] - df_master['date_of_birth']).dt.days / 365.25

# Drop columns that leak info, aren't useable, or extremely sparse
cols_to_drop = [
    'player_name',
    'player_slug',
    'player_image_url',
    'competition_name',
    'team_name',
    'joined',
    'contract_expires',
    'date_of_birth',
    'citizenship',
    'name_in_home_country',
    'date_of_death',
    'on_loan_from_club_id',
    'on_loan_from_club_name',
]
df_master = df_master.drop(columns=[c for c in cols_to_drop if c in df_master.columns])

# Encode categorical columns
df_master = pd.get_dummies(df_master, columns=['foot', 'is_eu'], sparse=True)

# Save model-ready dataset
df_master.to_csv(output_path, index=False)
print("Saved model-ready dataset")