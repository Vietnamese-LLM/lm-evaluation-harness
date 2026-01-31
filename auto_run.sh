#!/bin/bash
for step in {1000..8000..1000}
do
    formatted_step=$(printf "%07d" $step)
    
    echo "========================================================"
    echo "Starting evaluation for checkpoint: VLM-iter_${formatted_step}"
    echo "========================================================"

    lm_eval --model vllm \
        --model_args pretrained=Elfsong/VLM-iter_${formatted_step},tensor_parallel_size=1,dtype=bfloat16,gpu_memory_utilization=0.9,data_parallel_size=8 \
        --tasks vmlu,include_base_44_vietnamese,include_base_44_chinese,include_base_44_french,include_base_44_german,include_base_44_japanese,include_base_44_korean,include_base_44_malay,include_base_44_russian,vnhsge,vietnews,viquad,zalo_math,mmlu,lambada,hellaswag,truthfulqa_mc1,truthfulqa_mc2,truthfulqa_gen,arc_easy,arc_challenge,xcopa,belebele_vie_Latn,belebele_zho_Hans,belebele_eng_Latn,belebele_fra_Latn,belebele_jpn_Jpan,belebele_deu_Latn,belebele_zsm_Latn,belebele_kor_Hang,belebele_rus_Cyrl \
        --batch_size auto \
        --wandb_args project=vlm-eval,name=eval_iter_${formatted_step}
done
