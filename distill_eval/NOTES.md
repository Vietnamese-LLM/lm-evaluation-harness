# 1.7B Distillation Evaluation

## Setup
```bash
conda create -n vn_eval_env python=3.10 -y && conda activate vn_eval_env
cd /home/huynh37/lm-evaluation-harness
pip install -e . && pip install "lm_eval[vllm]" validators accelerate transformers sentencepiece peft
```

## Configure
Edit the `# Config` section at the top of `compare.sh` — set model paths, names, and `RESULTS_DIR`.

## Run
```bash
conda activate vn_eval_env && cd /home/huynh37/lm-evaluation-harness/distill_eval

bash compare.sh both      # interactive: runs both models + saves report
sbatch compare.sbatch     # SLURM (recommended), logs → logs/compare_<JOBID>.log

bash compare.sh baseline  # baseline only
bash compare.sh exp       # experimental only
bash compare.sh report    # re-generate report from existing results
```

## Files
```
distill_eval/
├── NOTES.md              ← this file
├── compare.sh            ← main script, configure model paths here
├── compare.sbatch        ← SLURM wrapper
├── aggregate_results.py  ← generates comparison table from result JSONs
└── results/
    ├── SUMMARY.md        ← summary of completed runs
    ├── 500vs800/         ← baseline@500 vs skip_lm_loss@800
    └── 500vs500/         ← baseline@500 vs skip_lm_loss@500
```

## Notes
- `DATA_PARALLEL_SIZE=1` — vLLM v1 doesn't support offline DP for dense models
- `GPU_MEMORY_UTILIZATION=0.6` — lowered from 0.9 to avoid OOM during wikitext logprobs
- Use `vmlu_final_v2`, not `vmlu_final` — `Elfsong/VMLU_Final` doesn't exist on HuggingFace
