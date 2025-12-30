import os

scripts = [
    "scripts/preprocess.py"
    "scripts/preprocess_player_profiles.py",
    "scripts/preprocess_player_performances.py",
    "scripts/preprocess_market_value.py",
    "scripts/preprocess_master_dataset.py",
    "scripts/merge_datasets.py",
]
 
total_steps = len(scripts)

for step, script in enumerate(scripts, start=1):
    print(f"Step {step}/{total_steps}: Running {script} ...")
    os.system(f"python {script}")

print("Done preprocessing and merging raw datasets. Next step: Make features dataset.")
