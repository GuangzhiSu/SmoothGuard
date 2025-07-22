export CUDA_VISIBLE_DEVICES=5

nohup python3 /home/gs285/MLLM-defense-using-random-smoothing/eval/model_vqa_loader_normal.py \
    --hf-model-name Salesforce/blip2-opt-2.7b \
    --image-folder /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/val2014 \
    --question-file /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/llava_pope_test.jsonl \
    --answers-file /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/answers/blip2-opt-2.7b-normal.jsonl \
    --device cuda \
    --max-new-tokens 256
    > /home/gs285/MLLM-defense-using-random-smoothing/eval/pope/blip2-opt-2.7b-normal.log 2>&1 &