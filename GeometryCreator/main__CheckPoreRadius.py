import os
import glob
import csv
import numpy as np
import porespy as ps
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
BASE_DIRECTORIES = ["../../simulations/Train_Danny_120_120_120_Pressure/"]
RAW_FILENAME = "domain.raw"
VOL_SHAPE = (120, 120, 120)
VOL_DTYPE = np.uint8

# Verification Parameters
R1, R2 = 2.0, 5.0
TARGET_PERCENT = 80.0

def plot_debug_lt(values, counts, r1, r2, real_pct, save_path):
    plt.figure(figsize=(10, 6))
    colors = ['navy' if r1 <= v <= r2 else 'skyblue' for v in values]
    plt.bar(values, counts, color=colors, edgecolor='black', alpha=0.7)
    plt.axvline(r1, color='green', linestyle='--', label=f'R1: {r1}')
    plt.axvline(r2, color='green', linestyle='--', label=f'R2: {r2}')
    plt.title(f'LT Distribution - Coverage: {real_pct:.2f}%')
    plt.xlabel('Radius [pixels]')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

overall_csv = "./overall_lt_verification.csv"
overall_rows = []

for base_dir in BASE_DIRECTORIES:
    print(f"Analyzing: {base_dir}")
    raw_files = glob.glob(os.path.join(base_dir, "**", RAW_FILENAME), recursive=True)
    
    dataset_rows = []
    
    for idx, path in enumerate(raw_files):
        folder_name = os.path.basename(os.path.dirname(path))
        
        # Load and process
        vol = np.fromfile(path, dtype=VOL_DTYPE).reshape(VOL_SHAPE)
        lt = ps.filters.local_thickness(vol)
        
        fluid_pixels = lt[lt > 0]
        total_vol = len(fluid_pixels)
        
        if total_vol == 0:
            continue

        # Stats and Verification
        values, counts = np.unique(fluid_pixels, return_counts=True)
        in_range = np.sum(fluid_pixels[(fluid_pixels >= R1) & (fluid_pixels <= R2)])
        real_pct = (len(fluid_pixels[(fluid_pixels >= R1) & (fluid_pixels <= R2)]) / total_vol) * 100.0
        
        row = {
            "file_idx": idx,
            "folder": folder_name,
            "lt_mean": np.mean(fluid_pixels),
            "lt_max": np.max(fluid_pixels),
            "pct_in_range": real_pct,
            "target_met": real_pct >= TARGET_PERCENT
        }
        dataset_rows.append(row)
        
        # Debug Plot for each sample
        plot_name = os.path.join(os.path.dirname(path), "lt_debug.png")
        plot_debug_lt(values, counts, R1, R2, real_pct, plot_name)

    # Save local CSV for the base directory
    if dataset_rows:
        local_csv = os.path.join(base_dir, "lt_verification.csv")
        with open(local_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=dataset_rows[0].keys())
            writer.writeheader()
            writer.writerows(dataset_rows)
            
        # Summary for overall report
        avg_pct = np.mean([r["pct_in_range"] for r in dataset_rows])
        overall_rows.append({
            "dataset": base_dir,
            "avg_coverage": avg_pct,
            "samples_count": len(dataset_rows)
        })

# Save final summary
with open(overall_csv, "w", newline="") as f:
    if overall_rows:
        writer = csv.DictWriter(f, fieldnames=overall_rows[0].keys())
        writer.writeheader()
        writer.writerows(overall_rows)

print("Analysis complete.")