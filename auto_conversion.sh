# Login to Hugging Face
hf auth login

source /opt/venv/bin/activate
export PYTHONPATH="/duyntc2/megatron-bridge/src:/duyntc2/megatron-bridge/3rdparty/Megatron-LM"
export WANDB_MODE=offline

for step in {500..10000..1000}
do
    formatted_step=$(printf "%07d" $step)

    echo "=================================================="
    echo "Converting checkpoint: iter_${formatted_step}"
    echo "=================================================="

    # Convert the Nemo checkpoint to HF
    python -m torch.distributed.run --nproc-per-node=1 /duyntc2/megatron-bridge/examples/conversion/convert_checkpoints.py export \
        --hf-model /traininguser/megatron-bridge/weights/qwen3_32b \
        --megatron-path /traininguser/stage_2_training/megatron-bridge/nemo_experiments/qwen3_32b_24_node/checkpoints/iter_${formatted_step} \
        --hf-path ./iter_${formatted_step}

    hf upload Elfsong/VLM_stage_2_iter_${formatted_step} ./iter_${formatted_step}/
done
