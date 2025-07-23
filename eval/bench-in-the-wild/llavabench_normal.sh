#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

# python3 /home/gs285/MLLM-defense-using-random-smoothing/eval/model_vqa_loader_normal.py \
#     --hf-model-name Qwen/Qwen2.5-VL-7B-Instruct \
#     --image-folder /home/gs285/MLLM-defense-using-random-smoothing/eval/bench-in-the-wild/images \
#     --question-file /home/gs285/MLLM-defense-using-random-smoothing/eval/bench-in-the-wild/questions.jsonl \
#     --answers-file /home/gs285/MLLM-defense-using-random-smoothing/eval/bench-in-the-wild/answers/Qwen2.5-VL-7B-Instruct-normal.jsonl \
#     --device cuda \
#     --max-new-tokens 256

# mkdir -p /home/gs285/MLLM-defense-using-random-smoothing/eval/bench-in-the-wild/reviews

python3 /home/gs285/MLLM-defense-using-random-smoothing/eval/bench-in-the-wild/eval_wildbench_qwen.py \
    --question /home/gs285/MLLM-defense-using-random-smoothing/eval/bench-in-the-wild/questions.jsonl \
    --context /home/gs285/MLLM-defense-using-random-smoothing/eval/bench-in-the-wild/context.jsonl \
    --rule /home/gs285/MLLM-defense-using-random-smoothing/eval/bench-in-the-wild/rule.json \
    --answer-list \
        /home/gs285/MLLM-defense-using-random-smoothing/eval/bench-in-the-wild/answers/answers_gpt4.jsonl \
        /home/gs285/MLLM-defense-using-random-smoothing/eval/bench-in-the-wild/answers/Qwen2.5-VL-7B-Instruct-normal.jsonl \
    --output \
        /home/gs285/MLLM-defense-using-random-smoothing/eval/bench-in-the-wild/reviews/Qwen2.5-reviews-normal.jsonl

python3 /home/gs285/MLLM-defense-using-random-smoothing/eval/bench-in-the-wild/summarize_gpt_review.py -f /home/gs285/MLLM-defense-using-random-smoothing/eval/bench-in-the-wild/reviews/Qwen2.5-reviews-normal.jsonl
