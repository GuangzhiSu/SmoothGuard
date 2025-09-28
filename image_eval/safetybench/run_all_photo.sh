
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6
# 模型与数据路径
MODEL_PATH="llava-hf/llava-1.5-7b-hf"
QUESTION_DIR="/home/gs285/MLLM-defense-using-random-smoothing/image_eval/safetybench/processed_questions"
IMAGE_DIR="/home/gs285/MLLM-defense-using-random-smoothing/image_eval/safetybench/imgs"  # 添加缺失的IMAGE_DIR
SCENARIO="01-Illegal_Activitiy"  # 如果你的文件名确实是错拼写，请改回 01-Illegal_Activitiy

# 通用对抗图片
UNIVERSAL_IMAGE="/home/gs285/MLLM-defense-using-random-smoothing/image_eval/safetybench/prompt_constrained_32.bmp"

# 固定参数
NUM_COPY=5
MAX_NEW_TOKENS=128
VARIANT="SD"
DEVICE="cuda"

# # 生成 sigma 列表：0.05, 0.10, ..., 0.50
# SIGMAS=()
# for i in $(seq 5 5 50); do
#   SIGMAS+=("$(printf "%.2f" "$(bc -l <<< "$i/100")")")
# done

# TOTAL=${#SIGMAS[@]}
# echo "将运行 ${TOTAL} 组实验：sigma ∈ {${SIGMAS[*]}}，num_copy=${NUM_COPY}"

# for idx in "${!SIGMAS[@]}"; do
#   sigma="${SIGMAS[$idx]}"
#   OUT_DIR="questions_with_universal_sig${sigma}_nc${NUM_COPY}"
#   EVAL_DIR="eval_results_universal_sig${sigma}_nc${NUM_COPY}"

#   echo "==== [$((idx+1))/${TOTAL}] sigma=${sigma}, num_copy=${NUM_COPY} ===="

#   # 1) 推理
#   python3 /home/gs285/MLLM-defense-using-random-smoothing/image_eval/safetybench/model-safetybench-photo.py \
#     --hf-model-name "$MODEL_PATH" \
#     --question-dir "$QUESTION_DIR" \
#     --image-dir "$IMAGE_DIR" \
#     --output-dir "$OUT_DIR" \
#     --sigma "$sigma" \
#     --num-copy "$NUM_COPY" \
#     --device "$DEVICE" \
#     --max-new-tokens "$MAX_NEW_TOKENS" \
#     --variant "$VARIANT" \
#     --scenario "$SCENARIO" \
#     --universal-image "$UNIVERSAL_IMAGE"

  # 2) 评测
  # python C:/Users/Administrator/Desktop/MLLM-defense-using-random-smoothing-main/MLLM-defense-using-random-smoothing-main/evaluation.py \
  #   --output-dir "$OUT_DIR" \
  #   --eval-output-dir "$EVAL_DIR" \
  #   --scenario-list "$SCENARIO"
# done

# echo "全部 ${TOTAL} 组完成！"

############################################
# 追加：固定 sigma=0.10, num_copy=10 跑全部场景
############################################
FIXED_SIGMA="0.40"
FIXED_NUM_COPY="5"

ALL_SCENARIOS=("01-Illegal_Activitiy")

echo "开始跑全部场景：sigma=${FIXED_SIGMA}, num_copy=${FIXED_NUM_COPY}"

# 创建logs目录
mkdir -p /home/gs285/MLLM-defense-using-random-smoothing/image_eval/safetybench/logs

for SC in "${ALL_SCENARIOS[@]}"; do
  OUT_DIR="questions_with_universal_universal_sig${FIXED_SIGMA}_nc${FIXED_NUM_COPY}"
  EVAL_DIR="eval_results_universal_universal_sig${FIXED_SIGMA}_nc${FIXED_NUM_COPY}"

  echo "==== [ALL] Scenario=${SC} | sigma=${FIXED_SIGMA} | num_copy=${FIXED_NUM_COPY} ===="

  # 1) 推理
  nohup python3 /home/gs285/MLLM-defense-using-random-smoothing/image_eval/safetybench/model-safetybench-photo.py \
    --hf-model-name "$MODEL_PATH" \
    --question-dir "$QUESTION_DIR" \
    --image-dir "$IMAGE_DIR" \
    --output-dir "$OUT_DIR" \
    --sigma "$FIXED_SIGMA" \
    --num-copy "$FIXED_NUM_COPY" \
    --device "$DEVICE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --variant "$VARIANT" \
    --scenario "$SC" \
    --universal-image "$UNIVERSAL_IMAGE" \
    > "/home/gs285/MLLM-defense-using-random-smoothing/image_eval/safetybench/logs/model-safetybench-photo_${SC}_sig${FIXED_SIGMA}_nc${FIXED_NUM_COPY}.log" 2>&1 &
    
  # 2) 评测
  # python3 /home/gs285/MLLM-defense-using-random-smoothing/image_eval/safetybench/evaluation.py \
  #   --output-dir "$OUT_DIR" \
  #   --eval-output-dir "$EVAL_DIR" \
  #   --scenario-list "$SC"
  
  # 等待几秒再启动下一个进程，避免资源冲突
  sleep 5
done

echo "全部场景跑完！"