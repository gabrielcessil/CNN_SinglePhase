#!/bin/bash

# ---------------- SLURM Master Allocation ----------------
#SBATCH --job-name=Launch_chunk_000
#SBATCH --partition=close_cpu
#SBATCH --nodelist=node[008-020]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -t 7-00:00:00
#SBATCH --exclusive

echo "=========================================================="
echo "Sub-Launcher chunk_000 active."
echo "Scouted Node: $SLURMD_NODENAME"
echo "Submitting sample scripts to this exact node via sbatch..."
echo "=========================================================="


(
    cd "../../GradSimulations/Train_Danny_SphPore_120_120_120" || exit 1
    echo "Submitting Train_Danny_SphPore_120_120_120 to queue on $SLURMD_NODENAME..."
    sbatch --nodelist=$SLURMD_NODENAME ./run_grad.sh
)

(
    cd "../../GradSimulations/Train_Danny_SphPore_120_120_120" || exit 1
    echo "Submitting Train_Danny_SphPore_120_120_120 to queue on $SLURMD_NODENAME..."
    sbatch --nodelist=$SLURMD_NODENAME ./run_grad.sh
)

(
    cd "../../GradSimulations/Train_Danny_SphPore_120_120_120" || exit 1
    echo "Submitting Train_Danny_SphPore_120_120_120 to queue on $SLURMD_NODENAME..."
    sbatch --nodelist=$SLURMD_NODENAME ./run_grad.sh
)

(
    cd "../../GradSimulations/Train_Danny_SphPore_120_120_120" || exit 1
    echo "Submitting Train_Danny_SphPore_120_120_120 to queue on $SLURMD_NODENAME..."
    sbatch --nodelist=$SLURMD_NODENAME ./run_grad.sh
)

(
    cd "../../GradSimulations/Train_Danny_SphPore_120_120_120" || exit 1
    echo "Submitting Train_Danny_SphPore_120_120_120 to queue on $SLURMD_NODENAME..."
    sbatch --nodelist=$SLURMD_NODENAME ./run_grad.sh
)

echo "--> All sbatch commands issued. Sub-Launcher exiting."
