#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH --partition=el8-rpi
#SBATCH --gres=gpu:4
#SBATCH -t 06:00:00
#SBATCH -o train%j.out

. /gpfs/u/scratch/HPDM/shared/miniconda3/etc/profile.d/conda.sh
conda activate rl-fold

srun train.py -a -p "config/single-sequence-q.json" 1000
