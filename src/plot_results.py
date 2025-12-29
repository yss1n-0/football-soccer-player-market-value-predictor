import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Load DataFrame
df = pd.read_csv('data/processed/predictions_with_errors.csv')

# Remove invalid rows just in case
df = df[(df['value'] > 0) & (df['predicted_value'] > 0)]

# Actual vs Predicted (log)

# 1️. Actual vs Predicted Player Market Value (Log Scale)
# This scatter plot shows each player's actual market value on the x-axis and the model's predicted value on the y-axis.
# Both axes are logarithmic to clearly show errors across low- and high-value players.
# Points near the red dashed line indicate accurate predictions, while points above or below the line show over- or underestimation.
# This helps visualize overall model fit and detect systematic biases across player values.

plt.figure(figsize=(8, 8))

plt.scatter(df['value'], df['predicted_value'], alpha=0.3)

max_val = max(df['value'].max(), df['predicted_value'].max())
plt.plot([1, max_val], [1, max_val], 'r--', label='Perfect prediction')

plt.xscale('log')
plt.yscale('log')

# Format axes in millions of euros
plt.gca().xaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, _: f"€{x/1e6:.2f}M" if x>=1e5 else f"€{int(x):,}")
)
plt.gca().yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda y, _: f"€{y/1e6:.2f}M" if y>=1e5 else f"€{int(y):,}")
)

plt.xlabel("Actual Market Value (€)")
plt.ylabel("Predicted Market Value (€)")
plt.title("Actual vs Predicted Market Value (Log Scale)")

plt.tight_layout()
plt.show()

# Prediction Error vs Value

# 2. Prediction Error vs Actual Market Value
# This scatter plot shows the absolute prediction error (in euros) on the y-axis versus the actual market value on the x-axis.
# The horizontal red line at zero represents perfect prediction.
# Points above the line indicate overestimation, points below indicate underestimation.
# This plot highlights where the model tends to perform better or worse depending on the player’s value.

plt.figure(figsize=(8, 6))

plt.scatter(df['value'], df['prediction_error'], alpha=0.3)
plt.axhline(0, color='red', linestyle="--")

plt.xscale('log')

# Format axes in millions of euros
plt.gca().xaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, _: f"€{x/1e6:.2f}M" if x>=1e5 else f"€{int(x):,}")
)
plt.gca().yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda y, _: f"€{y/1e6:.2f}M")
)

plt.xlabel("Actual Market Value (€)")
plt.ylabel("Prediction Error (€)")
plt.title("Prediction Error vs Actual Market Value")

plt.tight_layout()
plt.show()

# Error Distribution (Absolute €)

# 3. Distribution of Prediction Errors (Absolute €)
# Histogram of prediction errors in euros, showing the frequency of different error magnitudes.
# The vertical red line at zero indicates perfect predictions.
# Most errors cluster near zero, representing typical model accuracy, while the tails reveal extreme over- or underestimations.
# This plot helps quantify the typical scale of errors and detect large outliers.

plt.figure(figsize=(8, 6))

# Multiply prediction_error by 100 for x-axis
plt.hist(df['prediction_error'], bins=30)
# Vertical line at 0
plt.axvline(0, color='red', linestyle='--', linewidth=2)

plt.gca().xaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, _: f"€{x/1e6:.1f}M")
)

plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

plt.xlabel("Predictions Error (€)")
plt.ylabel("Number of Players")
plt.title("Distribution of Prediction Errors (Absolute €)")

plt.tight_layout()
plt.show()

# Error Distribution Percent - Full Range (includes outliers)

# 4. Distribution of Prediction Errors (%) – Full Range (Including Outliers)
# Histogram of prediction errors expressed as percentages, including extreme outliers.
# The red vertical line at zero represents perfect predictions.
# Most players’ errors are close to 0%, but rare extreme values (e.g., 135%) appear as outliers.
# This gives a relative view of model accuracy across all players and highlights the spread and symmetry of errors.

plt.figure(figsize=(8, 6))

# Use symmetric bins around 0
max_error = max(abs(df['error_pct'].min()), abs(df['error_pct'].max()))
bins = np.linspace(-max_error, max_error, 30)  

plt.hist(df['error_pct'], bins=bins)

# Vertical line at 0
plt.axvline(0, color='red', linestyle='--', linewidth=2)

plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

plt.xlabel("Prediction Error (%)")
plt.ylabel("Number of Players")
plt.title("Distribution of Prediction Errors (%) – Full Range (Including Outliers)")

plt.xlim(-max_error, max_error)  # limit x-axis to symmetric range
plt.xticks(np.linspace(-max_error, max_error, 11))  # 11 ticks

plt.tight_layout()
plt.show()

# Error Distribution Percent - Main Distribution (±5%)

# 5. Distribution of Prediction Errors (%) – Main Distribution (±5%)
# Histogram of prediction errors in percentages focusing on the main ±5% range.
# The vertical red line at zero is the reference for perfect prediction.
# This zoomed-in view shows the bulk of the data without being distorted by extreme outliers.
# It clearly shows typical prediction performance and allows easy comparison of over- vs underestimation for most players.

plt.figure(figsize=(8, 6))

# Use symmetric bins around 0
max_error = max(abs(df['error_pct'].min()), abs(df['error_pct'].max()))
bins = np.arange(-5, 5 + 1, 1)  

plt.hist(df['error_pct'], bins=bins)

# Vertical line at 0
plt.axvline(0, color='red', linestyle='--', linewidth=2)

plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

plt.xlabel("Prediction Error (%)")
plt.ylabel("Number of Players")
plt.title("Distribution of Prediction Errors (%) – Main Distribution (±5%)")

plt.xlim(-5, 5) # limit x-axis to symmetric range
plt.xticks(np.arange(-5, 5 + 1, 1))

plt.tight_layout()
plt.show()
