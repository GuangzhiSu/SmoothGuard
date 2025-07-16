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
export CUDA_VISIBLE_DEVICES=0

python3 /home/gs285/MLLM-defense-using-random-smoothing/eval/model_vqa_loader.py \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/llava_pope_test.jsonl \
    --image-folder /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/val2014 \
    --answers-file /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python3 /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/eval_pope.py \
    --annotation-dir /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/coco \
    --question-file /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/llava_pope_test.jsonl \
    --result-file /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/answers/llava-v1.5-13b.jsonl
