#!/bin/bash

# 路径参数请按自己实际情况调整
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_PATH="/data1/yutong/MLLM-defense-using-random-smoothing/Qwen2.5-VL-7B-Instruct"
QUESTION_DIR="MM-SafetyBench/processed_questions"
IMAGE_DIR="MM-SafetyBench/imgs"
OUTPUT_DIR_NORMAL="questions_with_answers_normal"
OUTPUT_DIR_NEW="questions_with_answers_new"
SCENARIO_LIST="10-Legal_Opinion"   # 可以用空格分隔多个

# 1. 运行不扰动版
echo "== 正常推理 =="
python model-safetybench-normal.py \
  --hf-model-name "$MODEL_NAME" \
  --question-dir "$QUESTION_DIR" \
  --image-dir "$IMAGE_DIR" \
  --output-dir "$OUTPUT_DIR_NORMAL" \
  --variant SD

# 2. 运行扰动版
echo "== 随机平滑推理 =="
python model-safetybench-new.py \
  --hf-model-name "$MODEL_PATH" \
  --question-dir "$QUESTION_DIR" \
  --image-dir "$IMAGE_DIR" \
  --output-dir "$OUTPUT_DIR_NEW" \
  --sigma 0.1 \
  --num-copy 10 \
  --device cuda \
  --max-new-tokens 128 \
  --variant SD

# 3. 评测普通版
echo "== 评测正常推理 =="
python evaluation.py --output-dir "$OUTPUT_DIR_NORMAL" --eval-output-dir eval_results_normal

# 4. 评测扰动版
echo "== 评测扰动推理 =="
python evaluation.py --output-dir "$OUTPUT_DIR_NEW" --eval-output-dir eval_results_new

echo "全部完成！"