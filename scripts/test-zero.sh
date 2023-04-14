#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH --partition=el8-rpi
#SBATCH --gres=gpu:4
#SBATCH -t 00:30:00
#SBATCH -o test-zero.out

. /gpfs/u/scratch/HPDM/shared/miniconda3/etc/profile.d/conda.sh
conda activate rl-fold

srun python train.py "config/single-sequence-zero.json" 2 -a
