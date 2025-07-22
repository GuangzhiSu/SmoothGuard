#!/usr/bin/bash
#SBATCH --job-name=pope-benchmark          # 作业名
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=8                       # 每个任务分配的 CPU 核数
#SBATCH --gres=gpu:1                       # 需要的 GPU 数量（对应你想用 0,1,2 三张卡）
#SBATCH --mem=32G                                # 内存总量
#SBATCH -e slurm-%j.err                          # 标准错误输出文件
#SBATCH -o slurm-%j.out                          # 标准输出文件
#SBATCH --partition=athena-genai                       # 分区名，按需替换
#SBATCH --account=gs285                           # 账户名，按需替换


# （如果你的集群不会自动设置 CUDA_VISIBLE_DEVICES，也可以手动指定）
export CUDA_VISIBLE_DEVICES=5

nohup python3 /home/gs285/MLLM-defense-using-random-smoothing/eval/model_vqa_loader_new.py \
    --hf-model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --image-folder /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/val2014 \
    --question-file /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/llava_pope_test.jsonl \
    --answers-file /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/answers/llava-v1.5-13b.jsonl \
    --sigma 0.1 \
    --num-copy 10 \
    --device cuda \
    --max-new-tokens 128
    > /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/llava-v1.5-13b.log 2>&1 &