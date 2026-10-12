#!/bin/bash
#SBATCH --job-name=your_script_name
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=1-4 # Optionally comma-separated list 32,64,128, etc.
# For more, see https://www.wm.edu/offices/it/services/researchcomputing/using/running_jobs_slurm/
# For Python usage, see https://www.wm.edu/offices/it/services/researchcomputing/using/software/python/

source ~/.bashrc
module load cuda/12.3

# Your environment containing pytorch, etc.
conda activate ~/envs/nndl_env

python your_script_name.py \
    --model cnn \
    --num_epochs 25 \
    --num_workers $SLURM_NTASKS \
    --num_hidden $SLURM_ARRAY_TASK_ID