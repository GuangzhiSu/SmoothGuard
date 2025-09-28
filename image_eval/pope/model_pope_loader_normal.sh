export CUDA_VISIBLE_DEVICES=5

# nohup python3 /home/gs285/MLLM-defense-using-random-smoothing/image_eval/model_vqa_loader_normal.py \
#     --hf-model-name llava-hf/llava-1.5-7b-hf \
#     --image-folder /home/gs285/MLLM-defense-using-random-smoothing/image_eval/pope/val2014 \
#     --question-file /home/gs285/MLLM-defense-using-random-smoothing/image_eval/pope/llava_pope_test.jsonl \
#     --answers-file /home/gs285/MLLM-defense-using-random-smoothing/image_eval/pope/answers/llava-Instruct-normal.jsonl \
#     --device cuda \
#     --max-new-tokens 256
#     > /home/gs285/MLLM-defense-using-random-smoothing/image_eval/pope/answers/llava-Instruct-normal.log 2>&1 &

python3 /home/gs285/MLLM-defense-using-random-smoothing/image_eval/pope/eval_pope.py \
    --annotation-dir /home/gs285/MLLM-defense-using-random-smoothing/image_eval/pope/coco \
    --question-file /home/gs285/MLLM-defense-using-random-smoothing/image_eval/pope/llava_pope_test.jsonl \
    --result-file /home/gs285/MLLM-defense-using-random-smoothing/image_eval/pope/answers/llava_on_pope/llava-Instruct-normal.jsonl
