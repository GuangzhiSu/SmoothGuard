#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=5

python3 /home/gs285/MLLM-defense-using-random-smoothing/eval/model_vqa_loader_new.py \
    --hf-model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --image-folder /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/val2014 \
    --question-file /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/llava_pope_test.jsonl \
    --answers-file /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/answers/Qwen2.5-VL-7B-Instruct-new.jsonl \
    --sigma 0.1 \
    --num-copy 10 \
    --device cuda \
    --max-new-tokens 128

python3 /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/eval_pope.py \
    --annotation-dir /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/coco \
    --question-file /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/llava_pope_test.jsonl \
    --result-file /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/answers/Qwen2.5-VL-7B-Instruct-new.jsonl
