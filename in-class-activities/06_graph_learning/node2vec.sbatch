#!/bin/bash
#SBATCH --job-name=node2vec
#SBATCH --output=node2vec.out
#SBATCH --error=node2vec.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=macs40123
#SBATCH --mem-per-cpu=30G

module load python cuda

python node2vec.py
