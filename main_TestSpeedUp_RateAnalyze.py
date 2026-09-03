import os
import re
from pathlib import Path
import pandas as pd

# Define datasets as a dictionary
main_folders = {
    "Bentheimer": "./Example_Bentheimer_2/",
    # "Berea": "./Example_Berea/"
}

def get_max_timestep_from_vis(folder_path):
    p = Path(folder_path)
    if not p.exists():
        return None

    vis_pattern = re.compile(r'^vis(\d+)$')
    max_ts = 0
    found_vis = False
    
    for item in p.iterdir():
        if item.is_dir():
            match = vis_pattern.match(item.name)
            if match:
                found_vis = True
                ts = int(match.group(1))
                if ts > max_ts:
                    max_ts = ts
                    
    return max_ts if found_vis else None

# ==============================================================================
# DATA EXTRACTION
# ==============================================================================
results = []

for dataset_name, dataset_path in main_folders.items():
    dataset_p = Path(dataset_path)
    if not dataset_p.exists():
        continue
        
    for sample_folder in dataset_p.glob("Sample_*"):
        sample_name = sample_folder.name
        
        run_dir = sample_folder / "lbpm_run"
        started_dir = sample_folder / "lbpm_started_run"
        
        ts_standard = get_max_timestep_from_vis(run_dir)
        ts_started = get_max_timestep_from_vis(started_dir)
        
        results.append({
            "Dataset": dataset_name,
            "Sample": sample_name,
            "Standard_Timesteps": ts_standard,
            "NN_Started_Timesteps": ts_started
        })

# ==============================================================================
# FORMAT AND SAVE
# ==============================================================================
df = pd.DataFrame(results)

# Optional: Calculate Speedup
df['Speedup_Ratio'] = df['Standard_Timesteps'] / df['NN_Started_Timesteps']

print("\nConvergence Timesteps Comparison:")
print("-" * 75)
print(df.to_string(index=False))
print("-" * 75)

df.to_csv("timesteps_comparison.csv", index=False)
print("\nResults saved to timesteps_comparison.csv")