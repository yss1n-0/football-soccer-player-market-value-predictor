import os
import pandas as pd
import numpy as np
import joblib

# Round predicted values like Transfermarkt
def round_market_value(val):
    if val < 1_000_000:
        return round(val, -3)
    elif val < 10_000_000:
        return round(val, -5)
    else:
        return round(val, -6)

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', 'models', 'lgb_market_value_model.pkl')
feature_list_path = os.path.join(script_dir, '..', 'models', 'features.txt')
data_path = os.path.join(script_dir, '..', 'data', 'processed', 'features_dataset.csv')
output_path = os.path.join(script_dir, '..', 'data', 'processed', 'predictions.csv')

# Turn off scientific notation and force commas
pd.options.display.float_format = '{:,.0f}'.format

# Load model
model = joblib.load(model_path)

# Load training feature list correctly
with open(feature_list_path, 'r') as f:
    trained_features = [line.strip() for line in f.readlines()]  # <-- readlines(), not readline()

# Load data
df = pd.read_csv(data_path)

# Drop columns not used in training
cols_to_drop = [
    'value',
    'player_id',
    'team_id',
    'date_unix',
    'season_name',
    'season_start_year',
    'competition_id',
    'current_club_id',
    'current_club_name',
    'place_of_birth',
    'country_of_birth',
    'career_goals',
    'career_assists',
    'career_goals_contrib',
    'career_clean_sheets',
    'career_goals_conceded',
    'avg_goals_per_season',
    'avg_assists_per_season',
    'avg_goals_conceded_per_season',
    'avg_clean_sheets_per_season',
    'is_eu_False',
    'foot_Unknown',
    'team_avg_value',
]

X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Reindex to match training features
X = X.reindex(columns=trained_features, fill_value=0)

# Predict
y_pred_log = model.predict(X)
df['predicted_value'] = np.expm1(y_pred_log)

# Round values like Transfermarkt
df['predicted_value'] = df['predicted_value'].apply(round_market_value)

# Handle duplicates
df = df.groupby(['player_id', 'season_name', 'season_start_year', 'date_unix']).agg({
    'predicted_value': 'max',   # Take the largest prediction
    'value': 'first',
    'age': 'first',
    'minutes_played': 'sum'
}).reset_index()


# Save predictions
df[['player_id', 'season_name', 'season_start_year', 'date_unix', 'predicted_value']].to_csv(output_path, index=False)

# Print prediction description
print(df['predicted_value'].describe())

print("Predictions saved to:", output_path)
print(df[['player_id', 'season_name', 'season_start_year', 'date_unix', 'predicted_value']].head(10))
