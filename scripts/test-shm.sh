#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH --partition=el8-rpi
#SBATCH --gres=gpu:4
#SBATCH -t 00:30:00
#SBATCH -o test-shm.out

. /gpfs/u/scratch/HPDM/shared/miniconda3/etc/profile.d/conda.sh
conda activate rl-fold

srun train.py "config/shm-test.json" 21 -i 0 -c
