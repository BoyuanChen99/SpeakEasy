#!/bin/bash

#SBATCH -q ebrainccs
#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=71:59:59
#SBATCH --mem=50GB
#SBATCH --job-name=seJBgcg

module purge

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate cipherword

# Lead to huggingface cache
export HF_HOME='/scratch/bc3194/huggingface_cache'

# Original parameters and call
frameworks=("baseline_dr" "speakeasy_dr" "baseline_gcg" "speakeasy_gcg" "baseline_tap" "speakeasy_tap")
# MODEL="vllm:meta-llama/Llama-3.3-70B-Instruct"
# for framework in "${frameworks[@]}"; do
#     echo "Running framework: $framework"
#     python3 run_frameworks.py \
#         --data-dir data/med-safety-bench/data.json \
#         --save-dir results/ \
#         --model $MODEL \
#         --frameworks "$framework"
# done

# Call python script with the model variable
MODEL="vllm:tiiuae/falcon-10B-instruct"
for framework in "${frameworks[@]}"; do
    echo "Running framework: $framework"
    python3 run_frameworks.py \
        --data-dir data/jailbreakbench/data.json \
        --save-dir results/ \
        --model $MODEL \
        --frameworks "$framework"
done