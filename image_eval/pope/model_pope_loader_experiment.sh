export CUDA_VISIBLE_DEVICES=4

nohup python3 /home/gs285/MLLM-defense-using-random-smoothing/image_eval/model_vqa_loader_experiment.py \
    --hf-model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --image-folder /home/gs285/MLLM-defense-using-random-smoothing/image_eval/pope/val2014 \
    --question-file /home/gs285/MLLM-defense-using-random-smoothing/image_eval/pope/llava_pope_test.jsonl \
    --answers-file /home/gs285/MLLM-defense-using-random-smoothing/image_eval/pope/answers/Qwen2.5-VL-7B-Instruct-experiment.jsonl \
    --sigma-start 0.05 \
    --sigma-end 0.5 \
    --sigma-step 0.05 \
    --device cuda \
    --max-new-tokens 128 \
    --subset-size 200 \
    > /home/gs285/MLLM-defense-using-random-smoothing/image_eval/pope/Qwen2.5-VL-7B-Instruct-experiment.log 2>&1 &