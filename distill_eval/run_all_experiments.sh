#!/bin/bash
# run_all_experiments.sh — Run both 500vs500 and 500vs800 comparisons.
#
# Usage:
#   bash run_all_experiments.sh [--overwrite] [all|baseline|500vs500|500vs800|exp-800|report]
#
#   --overwrite   Delete existing result dirs before running (clean slate)
#
# Examples:
#   bash run_all_experiments.sh                      # run everything, keep old results
#   bash run_all_experiments.sh --overwrite          # run everything, wipe old results first
#   bash run_all_experiments.sh --overwrite 500vs500 # wipe + rerun 500vs500 only
#   bash run_all_experiments.sh exp-800              # rerun ONLY skip_lm_loss@800 (keeps baseline)

set -e

# ── Parse --overwrite flag ────────────────────────────────────────────────────
OVERWRITE=false
if [[ "${1:-}" == "--overwrite" ]]; then
    OVERWRITE=true
    shift
fi

# ── Shared model ──────────────────────────────────────────────────────────────
BASELINE_PATH="/home/traininguser/thuatnh/distilled_megatron_bridge/distill_1.7B_converted/iter_0000500"
BASELINE_NAME="baseline_500"

# ── Experimental models ───────────────────────────────────────────────────────
EXP_500_PATH="/home/traininguser/thuatnh/distilled_megatron_bridge/distill_1.7B_converted_skip_lm_loss_true/iter_0000500"
EXP_500_NAME="skip_lm_loss_500"

EXP_800_PATH="/home/traininguser/thuatnh/distilled_megatron_bridge/distill_1.7B_converted_skip_lm_loss_true/iter_0000800"
EXP_800_NAME="skip_lm_loss_800"

# ── Results dirs ──────────────────────────────────────────────────────────────
RESULTS_500VS500="./results/500vs500"
RESULTS_500VS800="./results/500vs800"

# ── vLLM settings ─────────────────────────────────────────────────────────────
TENSOR_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=1
DTYPE="bfloat16"
GPU_MEMORY_UTILIZATION=0.6
BATCH_SIZE="auto"

# ── Tasks ─────────────────────────────────────────────────────────────────────
STAGE1_TASKS="lambada,wikitext,hellaswag"
STAGE2_TASKS="vmlu,vmlu_final_v2,include_base_44_vietnamese,vnhsge,vietnews,viquad,zalo_math,mmlu"
STAGE3_TASKS="truthfulqa_mc1,truthfulqa_mc2,truthfulqa_gen,arc_easy,arc_challenge,xcopa"

VLLM_ARGS="tensor_parallel_size=$TENSOR_PARALLEL_SIZE,dtype=$DTYPE,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,data_parallel_size=$DATA_PARALLEL_SIZE"

# ─────────────────────────────────────────────────────────────────────────────

clean_model_dir() {
    local NAME="$1" OUT="$2"
    if $OVERWRITE && [ -d "$OUT/$NAME" ]; then
        echo "--- Removing old results: $OUT/$NAME"
        rm -rf "$OUT/$NAME"
    fi
}

run_model() {
    local MODEL="$1" NAME="$2" OUT="$3"
    clean_model_dir "$NAME" "$OUT"
    echo "=== Evaluating: $NAME ($MODEL) ==="
    mkdir -p "$OUT/$NAME/stage_1" "$OUT/$NAME/stage_2" "$OUT/$NAME/stage_3"
    lm_eval --model vllm --model_args pretrained=$MODEL,$VLLM_ARGS \
        --tasks $STAGE1_TASKS --batch_size $BATCH_SIZE \
        --output_path "$OUT/$NAME/stage_1" --log_samples
    lm_eval --model vllm --model_args pretrained=$MODEL,$VLLM_ARGS \
        --tasks $STAGE2_TASKS --batch_size $BATCH_SIZE \
        --output_path "$OUT/$NAME/stage_2" --log_samples
    lm_eval --model vllm --model_args pretrained=$MODEL,$VLLM_ARGS \
        --tasks $STAGE3_TASKS --batch_size $BATCH_SIZE \
        --output_path "$OUT/$NAME/stage_3" --log_samples
    echo "✓ $NAME done → $OUT/$NAME"
}

gen_report() {
    local RESULTS_DIR="$1"
    echo "=== Generating report for $RESULTS_DIR ==="
    python aggregate_results.py --dir "$RESULTS_DIR" --save
    echo "✓ Report saved to $RESULTS_DIR/comparison_report.txt"
}

run_baseline_for() {
    # Reuse baseline results if already present in target dir (and not overwriting), else run it.
    local TARGET_DIR="$1"
    if ! $OVERWRITE && [ -d "$TARGET_DIR/$BASELINE_NAME" ]; then
        echo "--- Baseline already exists at $TARGET_DIR/$BASELINE_NAME, skipping re-run."
    else
        run_model "$BASELINE_PATH" "$BASELINE_NAME" "$TARGET_DIR"
    fi
}

case "${1:-all}" in
    baseline)
        run_model "$BASELINE_PATH" "$BASELINE_NAME" "$RESULTS_500VS500"
        # Symlink or copy baseline into 500vs800 to avoid re-running
        if [ ! -d "$RESULTS_500VS800/$BASELINE_NAME" ]; then
            mkdir -p "$RESULTS_500VS800"
            cp -r "$RESULTS_500VS500/$BASELINE_NAME" "$RESULTS_500VS800/$BASELINE_NAME"
            echo "✓ Baseline copied to $RESULTS_500VS800/$BASELINE_NAME"
        fi
        ;;
    500vs500)
        run_baseline_for "$RESULTS_500VS500"
        run_model "$EXP_500_PATH" "$EXP_500_NAME" "$RESULTS_500VS500"
        gen_report "$RESULTS_500VS500"
        ;;
    500vs800)
        run_baseline_for "$RESULTS_500VS800"
        run_model "$EXP_800_PATH" "$EXP_800_NAME" "$RESULTS_500VS800"
        gen_report "$RESULTS_500VS800"
        ;;
    exp-800)
        # Re-run only the iter-800 experimental model (baseline kept as-is)
        echo ">>> Re-running skip_lm_loss @ iter 800 only"
        rm -rf "$RESULTS_500VS800/$EXP_800_NAME"
        run_model "$EXP_800_PATH" "$EXP_800_NAME" "$RESULTS_500VS800"
        gen_report "$RESULTS_500VS800"
        ;;
    report)
        gen_report "$RESULTS_500VS500"
        gen_report "$RESULTS_500VS800"
        ;;
    all)
        echo ">>> Running 500vs500"
        run_model "$BASELINE_PATH" "$BASELINE_NAME" "$RESULTS_500VS500"

        echo ">>> Copying baseline to 500vs800 (avoid re-run)"
        mkdir -p "$RESULTS_500VS800"
        if $OVERWRITE || [ ! -d "$RESULTS_500VS800/$BASELINE_NAME" ]; then
            rm -rf "$RESULTS_500VS800/$BASELINE_NAME"
            cp -r "$RESULTS_500VS500/$BASELINE_NAME" "$RESULTS_500VS800/$BASELINE_NAME"
        fi

        echo ">>> Running skip_lm_loss @ iter 500"
        run_model "$EXP_500_PATH" "$EXP_500_NAME" "$RESULTS_500VS500"
        gen_report "$RESULTS_500VS500"

        echo ">>> Running skip_lm_loss @ iter 800"
        run_model "$EXP_800_PATH" "$EXP_800_NAME" "$RESULTS_500VS800"
        gen_report "$RESULTS_500VS800"

        echo ""
        echo "=== All experiments complete ==="
        echo "  Reports:"
        echo "    $RESULTS_500VS500/comparison_report.txt"
        echo "    $RESULTS_500VS800/comparison_report.txt"
        ;;
    *)
        echo "Usage: $0 [--overwrite] {all|baseline|500vs500|500vs800|exp-800|report}"
        exit 1
        ;;
esac
