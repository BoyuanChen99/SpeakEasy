#!/bin/bash

##SBATCH -q nvidia-xxl
#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=71:59:59
#SBATCH --mem=32GB
#SBATCH --job-name=SE-FC-HB

module purge

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate speakeasy

# Lead to huggingface cache
export HF_HOME='/scratch/bc3194/huggingface_cache'

# Call python script with the model variable

frameworks=("speakeasy_tap")
MODEL="vllm:tiiuae/Falcon3-10B-Instruct" # "openai:gpt-4o" # "qwen2:72b-instruct"

# harmbench
for framework in "${frameworks[@]}"; do
    echo "Running SpeakEasy framework: $framework"
    python3 run_frameworks.py \
        --data-dir data/harmbench/data.json \
        --save-dir results/ \
        --model $MODEL \
        --frameworks "$framework"
done
