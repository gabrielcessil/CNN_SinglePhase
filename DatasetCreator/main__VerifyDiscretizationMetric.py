import csv
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import porespy as ps
from utils import check_local_thickness, plot_lt_distribution

# Configure Matplotlib styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 11,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# --- CONFIGURATION ---
"""
BASE_DIRECTORIES = [
    "../../GradSimulations_Dist_40_5_55/DONE_Train_Danny_SphPore_120_120_120/",
    "../../GradSimulations_Dist_40_5_55/DONE_Valid_Danny_SphPore_120_120_120/",
    "../../GradSimulations_Dist_40_5_55/DONE_Test_Danny_SphPore_120_120_120/",
]
"""
BASE_DIRECTORIES = [
    "../../GradSimulations_Dist_40_5_55/DONE_Train_Oliveira_Bentheimer_120_120_120/",
]

#BASE_DIRECTORIES = [
#    "../../GradSimulations_Dist_40_5_55/DONE_Train_Oliveira_BereaUpperGray_120_120_120/",
#]

RAW_FILENAME = "domain.raw"
VOL_SHAPE = (120, 120, 120)
VOL_DTYPE = np.uint8

# Verification Limits
MIN_R = 5.0
MAX_R = 17.0
TARGET_PERCENT = 70.0

bins = 30

# --- GLOBAL ACCUMULATORS ---
global_dataset_summary = []
global_fluid_pixels = []
global_file_idx = 0

print("Extracting datasets...")

# --- DATA EXTRACTION ---
for base_dir in BASE_DIRECTORIES:
    raw_files = glob.glob(
        os.path.join(base_dir, "**", RAW_FILENAME), recursive=True
    )

    if not raw_files:
        print(f"Warning: No dataset found in {base_dir}. Skipping...")
        continue

    for raw_path in raw_files:
        parent_folder = os.path.basename(os.path.dirname(raw_path))
        base_dir_name = os.path.basename(os.path.normpath(base_dir))

        # Load raw volume
        vol = np.fromfile(raw_path, dtype=VOL_DTYPE).reshape(VOL_SHAPE)

        # Explore defined threshold
        success = check_local_thickness(
            im=vol,
            min_radius=MIN_R,
            max_radius=MAX_R,
            target_percentage=TARGET_PERCENT,
        )

        # Collect statistics
        lt = ps.filters.local_thickness(vol)
        fluid_pixels = lt[lt > 0]  # Where fluid cells are located
        mean_thickness = np.mean(fluid_pixels)
        std_thickness = np.std(fluid_pixels)

        global_dataset_summary.append(
            {
                "file_idx": global_file_idx,
                "base_dir": base_dir_name,
                "folder": parent_folder,
                "path": raw_path,
                "resolved": success,
                "mean_thickness": mean_thickness,
                "std_thickness": std_thickness,
            }
        )

        global_fluid_pixels.append(fluid_pixels)
        global_file_idx += 1


# --- OUTPUT & PLOTTING ---

# 1. Save Combined CSV
output_csv = "combined_local_thickness_summary.csv"
if global_dataset_summary:
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=global_dataset_summary[0].keys())
        writer.writeheader()
        writer.writerows(global_dataset_summary)
    print(f"Saved Combined CSV: {output_csv}")

# 2. Compute Combined Percentage & Generate Global Histogram
if global_fluid_pixels:
    combined_pixels = np.concatenate(global_fluid_pixels)

    # Calculate global percentage within target bounds
    in_range_count = np.sum((combined_pixels >= MIN_R) & (combined_pixels <= MAX_R))
    final_percentage = (in_range_count / len(combined_pixels)) * 100.0

    print(
        f"\nFinal Percentage inside [{MIN_R}, {MAX_R}] voxels: {final_percentage:.2f}%"
    )

    # Compute weights so total frequency equals 1.0 (100%)
    weights = np.ones_like(combined_pixels) / len(combined_pixels)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram with weights
    ax.hist(
        combined_pixels,
        bins=bins,
        weights=weights,
        color="dodgerblue",
        edgecolor="black",
        alpha=0.7,
    )

    # Set x-limits as requested [0, MAX_R + MIN_R]
    ax.set_xlim(0, MAX_R + MIN_R)

    # Formatting y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Percentage (%)")

    # Formatting general layout
    ax.set_title(
        f"Combined Local Thickness Distribution\n({len(BASE_DIRECTORIES)} Datasets, {len(global_dataset_summary)} Volumes Total)"
    )
    ax.set_xlabel("Local Thickness (voxels)")

    # Add vertical lines for verification limits (single legend entry)
    ax.axvline(
        MIN_R,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=rf"Limits ({int(MIN_R)}-{int(MAX_R)})",
    )
    ax.axvline(
        MAX_R,
        color="red",
        linestyle="dashed",
        linewidth=2,
    )
    ax.legend()

    # Save plot
    fig.tight_layout()
    fig.savefig("combined_thickness_histogram.pdf")
    plt.close(fig)

    print("Saved plot to combined_thickness_histogram.pdf")

print("\nBatch processing finished successfully.")