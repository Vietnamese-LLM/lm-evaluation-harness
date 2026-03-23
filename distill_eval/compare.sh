#!/bin/bash
# Usage: bash compare.sh [both|baseline|exp|report]
#   both    — run both models then save report  (default)
#   baseline — run baseline only
#   exp     — run experimental only
#   report  — re-generate report from existing results

set -e

# ── Config (edit here) ──────────────────────────────────────────────────────
BASELINE_PATH="/home/traininguser/thuatnh/distilled_megatron_bridge/distill_1.7B_converted/iter_0000500"
BASELINE_NAME="baseline_500"

EXP_PATH="/home/traininguser/thuatnh/distilled_megatron_bridge/distill_1.7B_converted_skip_lm_loss_true/iter_0000800"
EXP_NAME="skip_lm_loss_800"

RESULTS_DIR="./results"

TENSOR_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=1
DTYPE="bfloat16"
GPU_MEMORY_UTILIZATION=0.6
BATCH_SIZE="auto"

STAGE1_TASKS="lambada,wikitext,hellaswag"
STAGE2_TASKS="vmlu,vmlu_final_v2,include_base_44_vietnamese,vnhsge,vietnews,viquad,zalo_math,mmlu"
STAGE3_TASKS="truthfulqa_mc1,truthfulqa_mc2,truthfulqa_gen,arc_easy,arc_challenge,xcopa"
# ────────────────────────────────────────────────────────────────────────────

VLLM_ARGS="tensor_parallel_size=$TENSOR_PARALLEL_SIZE,dtype=$DTYPE,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,data_parallel_size=$DATA_PARALLEL_SIZE"

run_model() {
    local MODEL="$1" NAME="$2" OUT="$RESULTS_DIR/$NAME"
    echo "=== Evaluating: $NAME ($MODEL) ==="
    mkdir -p "$OUT/stage_1" "$OUT/stage_2" "$OUT/stage_3"
    lm_eval --model vllm --model_args pretrained=$MODEL,$VLLM_ARGS \
        --tasks $STAGE1_TASKS --batch_size $BATCH_SIZE --output_path "$OUT/stage_1" --log_samples
    lm_eval --model vllm --model_args pretrained=$MODEL,$VLLM_ARGS \
        --tasks $STAGE2_TASKS --batch_size $BATCH_SIZE --output_path "$OUT/stage_2" --log_samples
    lm_eval --model vllm --model_args pretrained=$MODEL,$VLLM_ARGS \
        --tasks $STAGE3_TASKS --batch_size $BATCH_SIZE --output_path "$OUT/stage_3" --log_samples
    echo "✓ $NAME done → $OUT"
}

case "${1:-both}" in
    baseline) run_model "$BASELINE_PATH" "$BASELINE_NAME" ;;
    exp)      run_model "$EXP_PATH" "$EXP_NAME" ;;
    both)
        run_model "$BASELINE_PATH" "$BASELINE_NAME"
        run_model "$EXP_PATH" "$EXP_NAME"
        python aggregate_results.py --dir "$RESULTS_DIR" --save
        ;;
    report)   python aggregate_results.py --dir "$RESULTS_DIR" ;;
    *)        echo "Usage: $0 {both|baseline|exp|report}"; exit 1 ;;
esac
