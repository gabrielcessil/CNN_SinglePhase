#!/bin/bash

#SBATCH --oversubscribe
#SBATCH --job-name=Perm
#SBATCH -t 7-0:00
#SBATCH -o perm_chunk_006_%j.out
#SBATCH -e perm_chunk_006_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=close_cpu
#SBATCH --mem=10G                  # RAM 64GB per a100.

export LBPM_VERSION="lbpm/cpu/lbpm_fork_parallelinitdebug_7c32db3"

module load $LBPM_VERSION

cd "lbpm_nn_run"
mpirun --oversubscribe -np 1 lbpm_permeability_simulator lbpm.db
