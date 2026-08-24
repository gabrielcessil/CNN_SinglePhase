#!/bin/bash

# ---------------- SLURM Job Settings ----------------
#SBATCH --job-name=Perm_Sample_00022
#SBATCH --partition=close_cpu
#SBATCH --nodelist=node[008-020]
#SBATCH --nodes=1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G

#SBATCH -t 7-00:00:00
#SBATCH -o perm_%j.out
#SBATCH -e perm_%j.err

grad_path="/home/gabriel.silveira/GRAD_LBM/grad-lbm"

module load conda/24.11.1
conda activate env_grad_lbm
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "--- Launching simulation for Sample_00022 ---"

$grad_path -i grad.ini

echo "--> Simulation for Sample_00022 finished."
