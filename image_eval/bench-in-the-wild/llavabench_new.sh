#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

# nohup python3 /home/gs285/MLLM-defense-using-random-smoothing/image_eval/model_vqa_loader_new.py \
#     --hf-model-name llava-hf/llava-1.5-7b-hf \
#     --image-folder /home/gs285/MLLM-defense-using-random-smoothing/image_eval/bench-in-the-wild/images \
#     --question-file /home/gs285/MLLM-defense-using-random-smoothing/image_eval/bench-in-the-wild/questions.jsonl \
#     --answers-file /home/gs285/MLLM-defense-using-random-smoothing/image_eval/bench-in-the-wild/answers/llava-Instruct-new.jsonl \
#     --sigma 0.1 \
#     --num-copy 10 \
#     --device cuda \
#     --max-new-tokens 256
#     > /home/gs285/MLLM-defense-using-random-smoothing/image_eval/bench-in-the-wild/answers/llava-Instruct-new.log 2>&1 &

mkdir -p /home/gs285/MLLM-defense-using-random-smoothing/image_eval/bench-in-the-wild/reviews

python3 /home/gs285/MLLM-defense-using-random-smoothing/image_eval/bench-in-the-wild/eval_wildbench_qwen.py \
    --question /home/gs285/MLLM-defense-using-random-smoothing/image_eval/bench-in-the-wild/questions.jsonl \
    --context /home/gs285/MLLM-defense-using-random-smoothing/image_eval/bench-in-the-wild/context.jsonl \
    --rule /home/gs285/MLLM-defense-using-random-smoothing/image_eval/bench-in-the-wild/rule.json \
    --answer-list \
        /home/gs285/MLLM-defense-using-random-smoothing/image_eval/bench-in-the-wild/answers/answers_gpt4.jsonl \
        /home/gs285/MLLM-defense-using-random-smoothing/image_eval/bench-in-the-wild/answers/llava-Instruct-new.jsonl \
    --output \
        /home/gs285/MLLM-defense-using-random-smoothing/image_eval/bench-in-the-wild/reviews/llava-reviews-new.jsonl

python3 /home/gs285/MLLM-defense-using-random-smoothing/image_eval/bench-in-the-wild/summarize_gpt_review.py -f /home/gs285/MLLM-defense-using-random-smoothing/image_eval/bench-in-the-wild/reviews/llava-reviews-new.jsonl
