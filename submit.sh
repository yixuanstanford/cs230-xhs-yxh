#!/bin/bash
#
#SBATCH -p serc
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G

ml python/3.9.0
ml opencv/4.5.5
ml py-pytorch/1.11.0_py39
ml py-numpy/1.20.3_py39
ml py-pandas/1.3.1_py39

python3 inference_image.py
