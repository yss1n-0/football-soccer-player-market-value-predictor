import os
import pandas as pd
import numpy as np

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))

features_path = os.path.join(
    script_dir, '..', 'data', 'processed', 'features_dataset.csv'
)
predictions_path = os.path.join(
    script_dir, '..', 'data', 'processed', 'predictions.csv'
)

output_all_path = os.path.join(
    script_dir, '..', 'data', 'processed', 'predictions_with_errors.csv'
)
# output_targets_path = os.path.join(
#     script_dir, '..', 'data', 'processed', 'top_transfer_targets.csv'
# )

# Turn off scientific notation and force commas
pd.options.display.float_format = '{:,.0f}'.format

# Load data
features_df = pd.read_csv(features_path)
pred_df = pd.read_csv(predictions_path)

# Merge predictions with original features
df = features_df.merge(
    pred_df,
    on=['player_id', 'season_start_year'],
    how='inner'
)

# Safety check
assert 'value' in df.columns, "Actual market value column missing!"
assert 'predicted_value' in df.columns, "Predicted value column missing!"

# Get rid of invalid rows
df = df[(df['value'] > 0) & (df['predicted_value'] > 0)]

# Error calculations
df['prediction_error'] = df['predicted_value'] - df['value']
df['error_pct'] = df['prediction_error'] / df['value']


# Save full merged dataset
df.to_csv(output_all_path, index=False)


# targets = df[
#     (df['prediction_error'] > 1_000_000) &      # undervalued by €1m+
#     (df['age'] <= 25) &                         # young players
#     (df['minutes_played'] > 900) &              # actually plays
#     (df['contract_remaining_years'] <= 2)       # realistic transfers
# ]

# targets = targets.sort_values(
#     by='prediction_error',
#     ascending=False
# )

# # Keep useful columns only
# target_cols = [
#     'player_id',
#     'season_name',
#     'season_start_year',
#     'date_unix',
#     'age',
#     'main_position_Attack',
#     'main_position_Midfield',
#     'main_position_Defender',
#     'main_position_Goalkeeper',
#     'minutes_played',
#     'contract_remaining_years',
#     'value',
#     'predicted_value',
#     'prediction_error',
#     'error_pct',
# ]

# target_cols = [c for c in target_cols if c in targets.columns]

# targets[target_cols].to_csv(output_targets_path, index=False)


print("Analysis complete\n")

print("\nSaved files:")
print(output_all_path)
# print(output_targets_path)
