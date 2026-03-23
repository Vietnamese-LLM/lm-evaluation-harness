#!/bin/bash
#SBATCH --job-name=distill_all
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=logs/distill_all_%j.log
#SBATCH --error=logs/distill_all_%j.err
#SBATCH --mem=200G

set -e
mkdir -p logs

echo "Job $SLURM_JOB_ID started: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vn_eval_env
cd /home/huynh37/lm-evaluation-harness/distill_eval

# Pass through --overwrite and/or mode argument, e.g.:
#   sbatch run_all.sbatch                   → run all, keep old results
#   sbatch --export=ALL,ARGS="--overwrite" run_all.sbatch  → wipe and rerun
bash run_all_experiments.sh ${ARGS:-} "${1:-all}"

echo "Finished: $(date)"
