#!/bin/bash
#SBATCH --job-name=distill_compare
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=logs/compare_%j.log
#SBATCH --error=logs/compare_%j.err
#SBATCH --mem=200G

set -e
mkdir -p logs

echo "Job $SLURM_JOB_ID started: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vn_eval_env
cd /home/huynh37/lm-evaluation-harness/distill_eval

bash compare.sh both

echo "Finished: $(date)"
