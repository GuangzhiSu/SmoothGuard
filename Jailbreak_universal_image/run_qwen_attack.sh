#!/bin/bash

# Qwen2.5-VL Visual Attack Runner Script

export CUDA_VISIBLE_DEVICES=4

echo "=== Qwen2.5-VL Visual Adversarial Attack ==="
echo ""

# Check if CUDA is available
if ! python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "Warning: CUDA not available. Using CPU mode."
    export CUDA_VISIBLE_DEVICES=""
else
    echo "CUDA is available."
fi

# Default parameters
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
GPU_ID=0
N_ITERS=5000
ALPHA=1
EPS=16
SAVE_DIR="results_llava_llama_v2_constrained_16"
TEMPLATE_IMG="adversarial_images/test_image.jpeg"
HARMFUL_CORPUS="harmful_corpus/derogatory_corpus.csv"
CONSTRAINED=False

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --n-iters)
            N_ITERS="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --eps)
            EPS="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --template-img)
            TEMPLATE_IMG="$2"
            shift 2
            ;;
        --harmful-corpus)
            HARMFUL_CORPUS="$2"
            shift 2
            ;;
        --constrained)
            CONSTRAINED=true
            shift
            ;;
        --test)
            echo "Running model test..."
            python qwen2_5_vl_test.py
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-name MODEL     HuggingFace model name (default: Qwen/Qwen2.5-VL-7B-Instruct)"
            echo "  --gpu-id ID           GPU device ID (default: 0)"
            echo "  --n-iters N           Number of iterations (default: 5000)"
            echo "  --alpha A             Step size (default: 1)"
            echo "  --eps E               Epsilon budget (default: 32)"
            echo "  --save-dir DIR        Output directory (default: qwen_output)"
            echo "  --template-img IMG    Template image path (default: adversarial_images/clean.jpeg)"
            echo "  --harmful-corpus CORPUS  Harmful corpus path (default: harmful_corpus/derogatory_corpus.csv)"
            echo "  --constrained         Use constrained attack (default: unconstrained)"
            echo "  --test                Run model test only"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python3 qwen2_5_vl_visual_attack.py"
CMD="$CMD --model-name $MODEL_NAME"
CMD="$CMD --gpu_id $GPU_ID"
CMD="$CMD --n_iters $N_ITERS"
CMD="$CMD --alpha $ALPHA"
CMD="$CMD --eps $EPS"
CMD="$CMD --save_dir $SAVE_DIR"
CMD="$CMD --template_img $TEMPLATE_IMG"
CMD="$CMD --harmful_corpus $HARMFUL_CORPUS"

if [ "$CONSTRAINED" = true ]; then
    CMD="$CMD --constrained"
fi

echo "Command: $CMD"
echo ""
echo "Starting attack..."
echo "Press Ctrl+C to stop"

# Run the attack
$CMD
