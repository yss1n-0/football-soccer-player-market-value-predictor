import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Path to the current directory this script is in
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the model-ready dataset
file_path = os.path.join(script_dir, '..', 'data', 'processed', 'model_ready_dataset.csv')

# Output path for the features dataset
output_path = os.path.join(script_dir, '..', 'data', 'processed', 'features_dataset.csv')

# Load DataFrame
df = pd.read_csv(file_path)
df = df.sort_values(['player_id', 'season_start_year'])

# Get goal contributions
df['goal_contributions'] = df['goals'] + df['assists']

# Career stats
df['career_goals'] = df.groupby('player_id')['goals'].cumsum()
df['career_assists'] = df.groupby('player_id')['assists'].cumsum()
df['career_goals_contrib'] =  df['career_goals'] + df['assists']
df['career_clean_sheets'] = df.groupby('player_id')['clean_sheets'].cumsum()
df['career_goals_conceded'] = df.groupby('player_id')['goals_conceded'].cumsum()

# Averages
df['avg_goals_per_season'] = df.groupby('player_id')['goals'].transform('mean')
df['avg_assists_per_season'] = df.groupby('player_id')['assists'].transform('mean')
df['avg_goals_contrib_per_season'] = df.groupby('player_id')['goal_contributions'].transform('mean')
df['avg_clean_sheets_per_season'] = df.groupby('player_id')['clean_sheets'].transform('mean')
df['avg_goals_conceded_per_season'] = df.groupby('player_id')['goals_conceded'].transform('mean')

# Avoid division by zero by replacing 0 minutes with NaN, then fill NaN with 0
minutes = df['minutes_played'].replace(0, pd.NA)

# Per 90 metrics
df['goals_per_90'] = (df['goals'] / (minutes / 90)).fillna(0)
df['assists_per_90'] = (df['assists'] / (minutes / 90)).fillna(0)
df['goals_contrib_per_90'] = (df['goal_contributions'] / (minutes / 90)).fillna(0)
df['goals_conceded_per_90'] = (df['goals_conceded'] / (minutes / 90)).fillna(0)

# Clean sheet rate
df['clean_sheet_rate'] = (df['clean_sheets'] / df['nb_on_pitch']).fillna(0)


# Square player's age to capture non-linear effects on value
df['age_squared'] = df['age'] ** 2

# How long has a player been playing in years
# Subtract how many years they've been playing until the current season
df['experience_years'] = df['season_start_year'] - df.groupby('player_id')['season_start_year'].transform('min')

# Team's total and average goals scored in season
# Being on a high performing team can affect market value
df['team_total_goals'] = df.groupby(['team_id', 'season_start_year'])['goals'].transform('sum')
df['team_avg_goals'] = df.groupby(['team_id', 'season_start_year'])['goals'].transform('mean')
df['team_avg_goals_per_player'] = df['team_total_goals'] / df.groupby(['team_id', 'season_start_year'])['player_id'].transform('nunique')

# Contract-related features
df['short_contract'] = (df['contract_remaining_years'] <= 1).astype(int)
df['contract_remaining_ratio'] = df['contract_remaining_years'] / df['contract_remaining_years'].max()

df['is_goalkeeper'] = (df['main_position'] == 'Goalkeepers').astype(int)

# Normalize goals and assists vs position averages
df['goals_vs_pos_avg'] = df['goals'] / df.groupby(['main_position', 'season_start_year'])['goals'].transform('mean')
df['assists_vs_pos_avg'] = df['assists'] / df.groupby(['main_position', 'season_start_year'])['assists'].transform('mean')
df['goal_contrib_vs_pos_avg'] = df['goal_contributions'] / df.groupby(['main_position', 'season_start_year'])['goal_contributions'].transform('mean')

# Replace infs or NaNs from division by zero
df[['goals_vs_pos_avg', 'assists_vs_pos_avg', 'goal_contrib_vs_pos_avg']] = \
    df[['goals_vs_pos_avg', 'assists_vs_pos_avg', 'goal_contrib_vs_pos_avg']].replace([float('inf'), -float('inf')], 0).fillna(0)

# Recent form
df['last_5_goals'] = df.groupby('player_id')['goals'].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
df['last_5_assists'] = df.groupby('player_id')['assists'].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
df['last_5_goals_contrib'] = df['last_5_goals'] + df['last_5_assists']

# Performance vs last season
df['goals_change_vs_last_season'] = df['goals'] - df.groupby('player_id')['goals'].shift(1)
df['assists_change_vs_last_season'] = df['assists'] - df.groupby('player_id')['goals'].shift(1)
df[['goals_change_vs_last_season', 'assists_change_vs_last_season']] = \
    df[['goals_change_vs_last_season', 'assists_change_vs_last_season']].fillna(0)

# Average teammate value
df['team_avg_value'] = df.groupby(['team_id', 'season_start_year'])['value'].transform('mean')

# Age bins
df['age_group'] = pd.cut(df['age'], bins=[15, 18, 21, 24, 28, 32, 40], labels=False)

# Hot transfer candidate
df['hot_transfer_candidate'] = ((df['contract_remaining_years'] <= 1) &
                                (df['age'] < 25) &
                                (df['avg_goals_contrib_per_season'] > 5)).astype(int)

# Numeric features to scale
numeric_features = [
    'age',
    'age_squared',
    'experience_years',
    'goals_per_90',
    'assists_per_90',
    'goals_contrib_per_90',
    'minutes_played',
    'career_goals',
    'career_assists',
    'career_goals_contrib',
    'avg_goals_per_season',
    'avg_assists_per_season',
    'avg_goals_contrib_per_season',
    'avg_clean_sheets_per_season',
    'avg_goals_conceded_per_season',
    'goals_conceded_per_90',
    'clean_sheet_rate',
    'contract_remaining_years',
    'contract_remaining_ratio',
    'team_total_goals',
    'team_avg_goals',
    'team_avg_goals_per_player',
    'goals_vs_pos_avg',
    'assists_vs_pos_avg',
    'goal_contrib_vs_pos_avg',
    'career_goals_conceded',
    'career_clean_sheets',
    'last_5_goals',
    'last_5_assists',
    'last_5_goals_contrib',
    'goals_change_vs_last_season',
    'assists_change_vs_last_season',
    'team_avg_value'
]

# Fill missing numeric values with 0 before scaling
df[numeric_features] = df[numeric_features].fillna(0)

# Scale numeric features
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Encode main position and position
df = pd.get_dummies(df, columns=['position', 'main_position', 'age_group'], sparse=True)

# Save the feature-engineered dataset
df.to_csv(output_path, index=False)
print("Saved feature-engineered dataset")