import os
import pandas as pd

# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to master_dataset.csv
file_path = os.path.join(script_dir, '..', 'data', 'processed', 'master_dataset.csv')

# Load dataset
df_master = pd.read_csv(file_path)

# Check column types
print("-- Check column types --")
print(df_master.dtypes)

# Quick stats for numeric columns
print("-- Quick stats for numeric columns --")
print(df_master.describe())

# Count unique values for each column
print("-- Count unique values for each column --")
print(df_master.nunique())

# Count missing values
print("-- Count missing values --")
print(df_master.isna().sum())