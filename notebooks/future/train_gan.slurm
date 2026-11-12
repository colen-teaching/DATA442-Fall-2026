#!/bin/bash
#SBATCH --job-name=train_gan
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00

source ~/.bashrc
module load cuda/12.3
conda activate ~/envs/nndl_env

python train_wgan.py --model WGAN-GP