#!/bin/bash
for step in {1000..2000..1000}
do
    formatted_step=$(printf "%07d" $step)
    
    echo "========================================================"
    echo "Starting evaluation for checkpoint: iter_${formatted_step}"
    echo "========================================================"

    lm_eval --model vllm \
        --model_args pretrained=/home/duyntc2/megatron-bridge/sft_scripts/sft_weights/packed/iter_${formatted_step},tensor_parallel_size=1,dtype=bfloat16,gpu_memory_utilization=0.9,data_parallel_size=8 \
        --tasks vmlu_final_v2 \
        --batch_size auto \
	--log_samples \
        --output_path "./evaluation_results/iter_${formatted_step}" \
        --wandb_args project=vlm-eval,name=eval_iter_stage_duy_${formatted_step}
done
