#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH --partition=el8-rpi
#SBATCH --gres=gpu:1
#SBATCH -t 00:05:00
#SBATCH -o visualize%j.out

. /gpfs/u/scratch/HPDM/shared/miniconda3/etc/profile.d/conda.sh
conda activate rl-fold

srun simulate.py -o deep-q \
    single-sequence-results.out models/single-sequence-q_model_iter_793.pth \
    /gpfs/u/scratch/HPDM/shared/pdb/pdb_mmcif/mmcif_files/2igd.cif A
