#!/bin/bash

echo "========================================================="
echo " MAIN LAUNCHER: Submitting all Sub-Launchers to SLURM"
echo "========================================================="

j1=$(sbatch --parsable  run_gradlbm_chunk_000.sh)
echo "Submitted Sub-Launcher run_gradlbm_chunk_000.sh (Job ID: $j1)"

echo "--> All Sub-Launchers submitted to the queue."
