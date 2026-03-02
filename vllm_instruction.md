# Vietamnese LLM Evaluation

## Assign an exclusive node from the Slurm cluster
```bash
srun --exclusive --gres=gpu:8 --time=04:00:00 --pty bash -i
```

## Enable the environment
```bash
cd /home/mingzhed/Project/lm-evaluation-harness
source .venv/bin/activate
```

## Run the Evaluation
```bash
# Edit this script to confirm the evalaution range.

# Stage 1 Evaluation
bash auto_eval_stage_1.sh

# Stage 2 Evaluation
bash auto_eval_stage_2.sh

# Stage 3 Evaluation
bash auto_eval_stage_3.sh
```
