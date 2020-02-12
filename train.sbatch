#!/bin/bash
#SBATCH -p mistral
#SBATCH -N1
#SBATCH --exclusive
#SBATCH --time=72:00:00

source ./venv/bin/activate

srun python src/train.py --load-model models/unet3d-1581270355
