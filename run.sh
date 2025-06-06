#!/bin/bash
source .venv/bin/activate
source .env
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    exit 1
fi
export HF_HOME=./cache
export HUGGINGFACE_HUB_CACHE=hub
export TRANSFORMERS_CACHE=transformers
export HF_DATASETS_CACHE=datasets
# Set environment variables for optimal H100 performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# Optional: Enable TF32 for better performance on H100
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
#python sd3_infer.py --controlnet_ckpt models/sd3.5_large_controlnet_depth.safetensors --controlnet_cond_image inputs/depth.png --prompt "studio ghibli style gymnast"
python sd3_infer.py --batch example_batch_canny.json
