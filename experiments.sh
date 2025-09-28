#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=0

python3 -u experiment.py \
    --hf-model-name "llava-hf/llava-1.5-7b-hf" \
    --subset-size 20 \
    --prompt-file harmful_corpus/rtp_prompts.jsonl \
    --output_file results_sigma \
    --gpu_id 0 \
