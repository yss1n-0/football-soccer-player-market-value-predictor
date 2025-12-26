import os
import pandas as pd
import numpy as np
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
df['career_goals_contrib'] =  df['career_goals'] + df['career_assists']
df['career_clean_sheets'] = df.groupby('player_id')['clean_sheets'].cumsum()
df['career_goals_conceded'] = df.groupby('player_id')['goals_conceded'].cumsum()

# Career stats until last season
df['career_goals_prev'] = df.groupby('player_id')['goals'].cumsum().shift(1).fillna(0)
df['career_assists_prev'] = df.groupby('player_id')['assists'].cumsum().shift(1).fillna(0)
df['career_goals_contrib_prev'] =  df['career_goals_prev'] + df['career_assists_prev']
df['career_clean_sheets_prev'] = df.groupby('player_id')['clean_sheets'].cumsum().shift(1).fillna(0)
df['career_goals_conceded_prev'] = df.groupby('player_id')['goals_conceded'].cumsum().shift(1).fillna(0)


# Averages
df['avg_goals_per_season'] = df.groupby('player_id')['goals'].transform('mean')
df['avg_assists_per_season'] = df.groupby('player_id')['assists'].transform('mean')
df['avg_goals_contrib_per_season'] = df.groupby('player_id')['goal_contributions'].transform('mean')
df['avg_clean_sheets_per_season'] = df.groupby('player_id')['clean_sheets'].transform('mean')
df['avg_goals_conceded_per_season'] = df.groupby('player_id')['goals_conceded'].transform('mean')

# Averages until last season
df['avg_goals_per_season_prev'] = df.groupby('player_id')['goals'].transform('mean').shift(1).fillna(0)
df['avg_assists_per_season_prev'] = df.groupby('player_id')['assists'].transform('mean').shift(1).fillna(0)
df['avg_goals_contrib_per_season_prev'] = df.groupby('player_id')['goal_contributions'].transform('mean').shift(1).fillna(0)
df['avg_clean_sheets_per_season_prev'] = df.groupby('player_id')['clean_sheets'].transform('mean').shift(1).fillna(0)
df['avg_goals_conceded_per_season_prev'] = df.groupby('player_id')['goals_conceded'].transform('mean').shift(1).fillna(0)


# Avoid division by zero by replacing 0 minutes with NaN, then fill NaN with 0
minutes = df['minutes_played'].replace(0, pd.NA)

# Per 90 metrics
df['minutes_nonzero'] = df['minutes_played'].replace(0, pd.NA)
df['goals_per_90_season'] = (df['goals'] / (df['minutes_nonzero'] / 90)).fillna(0)
df['assists_per_90_season'] = (df['assists'] / (df['minutes_nonzero'] / 90)).fillna(0)
df['goals_contrib_per_90_season'] = (df['goal_contributions'] / (df['minutes_nonzero'] / 90)).fillna(0)

df['goals_per_90_last_season'] = df.groupby('player_id')['goals_per_90_season'].shift(1).fillna(0)
df['assists_per_90_last_season'] = df.groupby('player_id')['assists_per_90_season'].shift(1).fillna(0)
df['goals_contrib_per_90_last_season'] = df.groupby('player_id')['goals_contrib_per_90_season'].shift(1).fillna(0)
df['minutes_last_season'] = df.groupby('player_id')['minutes_played'].shift(1).fillna(0)

# ---------- rolling last 3 seasons (exclude current season) ----------
df['goals_per_90_last3_avg'] = df.groupby('player_id')['goals_per_90_season'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()).fillna(0)
df['assists_per_90_last3_avg'] = df.groupby('player_id')['assists_per_90_season'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()).fillna(0)
df['goals_contrib_per_90_last3_avg'] = df.groupby('player_id')['goals_contrib_per_90_season'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()).fillna(0)
df['minutes_last3_avg'] = df.groupby('player_id')['minutes_played'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()).fillna(0)

# ---------- competition / league level aggregation ----------
# competition_id column exists — compute competition-season avg/median value (shifted so we don't leak)
df['competition_prev_avg_value'] = df.groupby('competition_id')['value'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)

# Also create competition historical median up to previous season to avoid leakage:
df['competition_prev_median_value'] = df.groupby('competition_id')['value'].transform(lambda x: x.shift(1).expanding().median()).fillna(0)

# ---------- player peak/previous max value ----------
df['max_value_prev_seasons'] = df.groupby('player_id')['value'].transform(lambda x: x.shift(1).cummax()).fillna(0)

# ---------- season-level trend feature ----------
df['season_year_offset'] = df['season_start_year'] - df['season_start_year'].min()

# ---------- remove temporary minutes_nonzero before saving ----------
df.drop(columns=['minutes_nonzero'], inplace=True)

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

# Exponentially weighted rolling goal contributions over last 10 games
df['ewm_goals_contrib'] = df.groupby('player_id')['goal_contributions'].transform(
    lambda x: x.ewm(span=10, adjust=False).mean()
)

# Performance vs last season
df['goals_change_vs_last_season'] = df['goals'] - df.groupby('player_id')['goals'].shift(1)
df['assists_change_vs_last_season'] = df['assists'] - df.groupby('player_id')['assists'].shift(1)
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

# Trusted-weighted performance
df['trusted_goals_contrib'] = (
    df['goals_contrib_per_90_season'] * np.log1p(df['minutes_played'])
)

# Prime-age performance
df['prime_age_factor'] = 1 - (abs(df['age'] - 26) / 10)
df['prime_age_factor'] = df['prime_age_factor'].clip(lower=0)

# Contract pressure performance
df['contract_pressure_score'] = (
    df['goals_contrib_per_90_season'] /
    (1 + df['contract_remaining_years'])
)

# Weighted goal contributions
df['weighted_goals_contrib'] = (
    df['career_goals_contrib_prev'] * 0.7 +
    df['avg_goals_contrib_per_season_prev'] * 0.3
)

# Minutes trend
df['minutes_change_vs_last_season'] = df['minutes_played'] - df.groupby('player_id')['minutes_played'].shift(1).fillna(0)

# Age x Position interaction
position_cols = [c for c in df.columns if c.startswith('main_position_')]
for pos in position_cols:
    df[f'{pos}_age'] = df[pos] * df['age']

# Encode main position and position
df = pd.get_dummies(df, columns=['position', 'main_position', 'age_group'], sparse=True)

# Position and age features

df['prime_attacker'] = ((df['main_position_Attack']==1) & (df['age'].between(20,26))).astype(int)
df['prime_midfielder'] = ((df['main_position_Midfield']==1) & (df['age'].between(22,28))).astype(int)
df['prime_defender'] = ((df['main_position_Defender']==1) & (df['age'].between(24,30))).astype(int)
df['prime_goalkeeper'] = ((df['main_position_Goalkeeper']==1) & (df['age'].between(27,33))).astype(int)



# Numeric features to scale
numeric_features = [
    # Age / experience
    'age',
    'age_squared',
    'experience_years',
    
    # Per 90 metrics
    'goals_per_90_season',
    'assists_per_90_season',
    'goals_contrib_per_90_season',
    'goals_per_90_last_season',
    'assists_per_90_last_season',
    'goals_contrib_per_90_last_season',
    'goals_per_90_last3_avg',
    'assists_per_90_last3_avg',
    'goals_contrib_per_90_last3_avg',
    
    # Minutes
    'minutes_played',
    'minutes_last_season',
    'minutes_last3_avg',
    'minutes_change_vs_last_season',
    
    # Career totals
    'career_goals',
    'career_assists',
    'career_goals_contrib',
    'career_clean_sheets',
    'career_goals_conceded',
    
    # Career prev / last season
    'career_goals_prev',
    'career_assists_prev',
    'career_goals_contrib_prev',
    'career_clean_sheets_prev',
    'career_goals_conceded_prev',
    
    # Averages
    'avg_goals_per_season',
    'avg_assists_per_season',
    'avg_goals_contrib_per_season',
    'avg_clean_sheets_per_season',
    'avg_goals_conceded_per_season',
    'avg_goals_per_season_prev',
    'avg_assists_per_season_prev',
    'avg_goals_contrib_per_season_prev',
    'avg_clean_sheets_per_season_prev',
    'avg_goals_conceded_per_season_prev',
    
    # Team / competition
    'team_total_goals',
    'team_avg_goals',
    'team_avg_goals_per_player',
    'team_avg_value',
    'competition_prev_avg_value',
    'competition_prev_median_value',
    
    # Performance / normalized
    'goals_vs_pos_avg',
    'assists_vs_pos_avg',
    'goal_contrib_vs_pos_avg',
    'ewm_goals_contrib',
    'goals_change_vs_last_season',
    'assists_change_vs_last_season',
    'trusted_goals_contrib',
    'weighted_goals_contrib',
    
    # Contract / pressure
    'contract_remaining_years',
    'contract_remaining_ratio',
    'prime_age_factor',
    'contract_pressure_score',
    
    # Peak value / trends
    'max_value_prev_seasons',
    'season_year_offset',
    
    # Clean sheets / conceded per 90
    'clean_sheet_rate',

    # Position
    'prime_attacker',
    'prime_midfielder',
    'prime_defender',
    'prime_goalkeeper',
]

# Fill missing numeric values with 0 before scaling
# df[numeric_features] = df[numeric_features].fillna(0)

# Scale numeric features
# df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Save the feature-engineered dataset
df.to_csv(output_path, index=False)
print("Saved feature-engineered dataset")