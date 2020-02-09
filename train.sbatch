#!/bin/bash
#SBATCH -p mistral
#SBATCH -N1
#SBATCH --exclusive
#SBATCH --time=24:00:00

source ./venv/bin/activate

srun python src/train.py
