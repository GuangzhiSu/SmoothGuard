#!/usr/bin/bash
#SBATCH --job-name=egoschema-benchmark          # 作业名
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=8                       # 每个任务分配的 CPU 核数
#SBATCH --gres=gpu:1                       # 需要的 GPU 数量（对应你想用 0,1,2 三张卡）
#SBATCH --mem=32G                                # 内存总量
#SBATCH -e slurm-%j.err                          # 标准错误输出文件
#SBATCH -o slurm-%j.out                          # 标准输出文件
#SBATCH --partition=athena-genai                       # 分区名，按需替换
#SBATCH --account=gs285                           # 账户名，按需替换


# （如果你的集群不会自动设置 CUDA_VISIBLE_DEVICES，也可以手动指定）
export CUDA_VISIBLE_DEVICES=0

python3 -u /home/gs285/mllm-defense-using-random-smoothing/experiment.py \
    --hf-model-name "llava-hf/llava-1.5-7b-hf" \
    --subset-size 20 \
    --prompt-file /home/gs285/mllm-defense-using-random-smooth/harmful_corpus/rtp_prompts.jsonl \
    --output_file /home/gs285/mllm-defense-using-random-smooth/results_sigma \
    --gpu_id 0 \
