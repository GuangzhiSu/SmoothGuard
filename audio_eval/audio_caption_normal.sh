export CUDA_VISIBLE_DEVICES=3
nohup python3 /home/gs285/MLLM-defense-using-random-smoothing/audio_eval/audio_caption_normal.py \
    --model-name Qwen/Qwen2-Audio-7B \
    --benchmark-csv /home/gs285/MLLM-defense-using-random-smoothing/audio_eval/CompA_order/CompA_order_benchmark.csv \
    --audio-folder /home/gs285/MLLM-defense-using-random-smoothing/audio_eval/CompA_order/CompA_order_files \
    --device cuda:0 \
    --max-samples 1000 \
    --output-file /home/gs285/MLLM-defense-using-random-smoothing/audio_eval/audio_caption_normal_results.json \
    > /home/gs285/MLLM-defense-using-random-smoothing/audio_eval/audio_caption_normal.log 2>&1 &