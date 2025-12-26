import os
import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Path to the current directory this script is in
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the features dataset
file_path = os.path.join(script_dir, '..', 'data', 'processed', 'features_dataset.csv')

# Load DataFrame
df = pd.read_csv(file_path)

# Target variable
TARGET = 'value'

# Drop rows with missing target
df = df.dropna(subset=[TARGET])

# Columns to exclude from training
cols_to_drop = [
    'value',
    'player_id',
    'team_id',
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
y = df[TARGET]
y_log = np.log1p(y)

# Time-based train/test split
split_year = 2020
train_mask = df['season_start_year'] < split_year
test_mask = df['season_start_year'] >= split_year

X_train, X_test = X[train_mask].copy(), X[test_mask].copy()
y_train, y_test = y_log[train_mask], y_log[test_mask]

# Drop low-variance columns
low_variance_cols = [c for c in X_train.columns if X_train[c].nunique() <= 2]
X_train = X_train.drop(columns=low_variance_cols)
X_test = X_test.drop(columns=low_variance_cols)
print(f"Dropped {len(low_variance_cols)} low-variance columns")
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

numeric_features = [
    # Age / experience
    'age', 'age_squared', 'experience_years',
    # Per 90 metrics
    'goals_per_90_season', 'assists_per_90_season', 'goals_contrib_per_90_season',
    'goals_per_90_last_season', 'assists_per_90_last_season', 'goals_contrib_per_90_last_season',
    'goals_per_90_last3_avg', 'assists_per_90_last3_avg', 'goals_contrib_per_90_last3_avg',
    # Minutes
    'minutes_played', 'minutes_last_season', 'minutes_last3_avg', 'minutes_change_vs_last_season',
    # Career totals
    'career_goals', 'career_assists', 'career_goals_contrib', 'career_clean_sheets', 'career_goals_conceded',
    # Career prev / last season
    'career_goals_prev', 'career_assists_prev', 'career_goals_contrib_prev',
    'career_clean_sheets_prev', 'career_goals_conceded_prev',
    # Averages
    'avg_goals_per_season', 'avg_assists_per_season', 'avg_goals_contrib_per_season', 'avg_clean_sheets_per_season',
    'avg_goals_conceded_per_season', 'avg_goals_per_season_prev', 'avg_assists_per_season_prev',
    'avg_goals_contrib_per_season_prev', 'avg_clean_sheets_per_season_prev', 'avg_goals_conceded_per_season_prev',
    # Team / competition
    'team_total_goals', 'team_avg_goals', 'team_avg_goals_per_player', 'team_avg_value', 'competition_prev_avg_value',
    'competition_prev_median_value',
    # Performance / normalized
    'goals_vs_pos_avg', 'assists_vs_pos_avg', 'goal_contrib_vs_pos_avg', 'ewm_goals_contrib',
    'goals_change_vs_last_season', 'assists_change_vs_last_season', 'trusted_goals_contrib',
    'weighted_goals_contrib',
    # Contract / pressure
    'contract_remaining_years', 'contract_remaining_ratio', 'prime_age_factor', 'contract_pressure_score',
    # Peak value / trends
    'max_value_prev_seasons', 'season_year_offset',
    # Clean sheets / conceded per 90
    'clean_sheet_rate',
    # Position
    'prime_attacker', 'prime_midfielder', 'prime_defender', 'prime_goalkeeper',
]
numeric_features = [c for c in numeric_features if c in X_train.columns]

# LGBM model
tscv = TimeSeriesSplit(n_splits=3)
lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)  # reduced verbose

param_grid_lgb = {
    'num_leaves': [31, 63],
    'learning_rate': [0.03, 0.05],
    'n_estimators': [300, 600],
    'min_child_samples': [5, 10, 20],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 1, 5],
    'reg_lambda': [0, 1, 5]
}

search_lgb = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_grid_lgb,
    n_iter=20,
    scoring='neg_mean_absolute_error',
    cv=tscv,
    n_jobs=-1,
    verbose=1,  # keep progress but not debug
    random_state=42
)
search_lgb.fit(X_train, y_train)
best_lgb = search_lgb.best_estimator_
print("Best LGB params:", search_lgb.best_params_)

y_pred_log = best_lgb.predict(X_test)
y_pred = np.expm1(y_pred_log)

mae = mean_absolute_error(np.expm1(y_test), y_pred)
r2 = r2_score(np.expm1(y_test), y_pred)
print(f"\nLGB MAE: €{mae:,.0f}")
print(f"LGB R²: {r2:.3f}")

# HGB model
scaler = StandardScaler()
X_train_hgb = X_train.copy()
X_test_hgb = X_test.copy()
X_train_hgb[numeric_features] = scaler.fit_transform(X_train_hgb[numeric_features])
X_test_hgb[numeric_features] = scaler.transform(X_test_hgb[numeric_features])

hgb = HistGradientBoostingRegressor(random_state=42, early_stopping=True)
param_grid = {
    'max_iter': [500, 700, 1000],
    'learning_rate': [0.05, 0.07, 0.1],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [20, 30, 50],
    'l2_regularization': [0, 1, 5]
}

search_hgb = RandomizedSearchCV(
    estimator=hgb,
    param_distributions=param_grid,
    n_iter=20,
    scoring='neg_mean_absolute_error',
    cv=3,
    n_jobs=1,
    verbose=2,
    random_state=42
)
search_hgb.fit(X_train_hgb, y_train)
best_hgb = search_hgb.best_estimator_
print("\nBest HGB params:", search_hgb.best_params_)

# Evaluate HGB
y_pred_log_hgb = best_hgb.predict(X_test_hgb)
y_pred_hgb = np.expm1(y_pred_log_hgb)
mae_hgb = mean_absolute_error(np.expm1(y_test), y_pred_hgb)
r2_hgb = r2_score(np.expm1(y_test), y_pred_hgb)
print(f"\nHGB MAE: €{mae_hgb:,.0f}")
print(f"HGB R²: {r2_hgb:.3f}")

# Baseline
baseline_pred = np.full(shape=len(y_test), fill_value=np.expm1(y_train.mean()))
baseline_mae = mean_absolute_error(np.expm1(y_test), baseline_pred)
print(f"\nBaseline MAE (mean prediction): €{baseline_mae:,.0f}")
print(f"MAE improvement: €{baseline_mae - mae:,.0f}")

# Feature importance
best_model = best_lgb if mae < mae_hgb else best_hgb
result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
feat_importance = pd.Series(result.importances_mean, index=X_train.columns).sort_values(ascending=False)
print("\nTop 15 important features:")
print(feat_importance.head(15))
