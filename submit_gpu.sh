#!/bin/bash
#
#SBATCH --job-name=cs230_4
#SBATCH -p serc
#SBATCH -c 8
#SBATCH -t 20:00:00
#SBATCH -G 4
ml python/3.9.0
ml opencv/4.5.5
ml py-pytorch/1.11.0_py39
ml py-numpy/1.20.3_py39
ml py-pandas/1.3.1_py39

python3 train.py
