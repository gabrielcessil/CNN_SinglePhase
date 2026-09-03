import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_permeability_comparison(sample_folder):
    run_csv = os.path.join(sample_folder, "lbpm_run", "Permeability.csv")
    started_csv = os.path.join(sample_folder, "lbpm_started_run", "Permeability.csv")
    
    # Check if files exist
    if not os.path.exists(run_csv) or not os.path.exists(started_csv):
        print(f"Error: Could not find Permeability.csv in one or both subfolders of {sample_folder}")
        return

    # Load data (Assuming standard LBPM headers, adjust column names if necessary)
    # Often it outputs 'time' and 'k' or 'permeability'. We'll strip whitespace from headers.
    df_run = pd.read_csv(run_csv)
    df_started = pd.read_csv(started_csv)
    
    df_run.columns = df_run.columns.str.strip()
    df_started.columns = df_started.columns.str.strip()

    # Identify time and permeability columns based on typical LBPM outputs
    time_col = df_run.columns[0]  # Usually the first column is timestep/time
    perm_col = df_run.columns[1]  # Usually the second column is the permeability tensor/value

    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot Standard Run
    plt.plot(df_run[time_col], df_run[perm_col], 
             label='Standard Run', color='#e74c3c', linewidth=2)
             
    # Plot NN-Started Run
    plt.plot(df_started[time_col], df_started[perm_col], 
             label='NN-Initiated Run', color='#16a085', linewidth=2, linestyle='--')

    plt.xlabel('Timestep')
    plt.ylabel('Permeability (LBM Units)')
    plt.title(f'Permeability Convergence: {os.path.basename(sample_folder)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(sample_folder, "permeability_convergence.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Allows passing the sample folder as a command-line argument
    parser = argparse.ArgumentParser(description="Plot Permeability vs Timestep")
    parser.add_argument(
        "--folder", 
        type=str, 
        default="./Example_Bentheimer_2/Sample_01",
        help="Path to the sample folder containing lbpm_run and lbpm_started_run"
    )
    args = parser.parse_args()
    
    plot_permeability_comparison(args.folder)