# Audio-Caption Matching Benchmark

This directory contains evaluation scripts for the CompA_order audio-caption matching benchmark. The task is to correctly match audio files with their corresponding captions, where the captions contain the same words but in different orders to reflect different audio compositions.

## Benchmark Structure

The benchmark consists of:
- **Audio files**: Different compositions of the same audio elements (e.g., animal sounds + human speech)
- **Captions**: Three different captions for each set, containing the same words but in different orders:
  - `pair_caption`: Original composition description
  - `reversed_pair_caption`: Reversed order description  
  - `triplet_caption`: Mixed/interleaved composition description

## Task Description

For each set of audio files, the model must:
1. Listen to each audio file
2. Generate a caption describing what it hears
3. Match the generated caption with the most similar ground truth caption from the available options
4. The task is evaluated as a multiple-choice question where the model selects the correct caption from the candidates

## Evaluation Scripts

### 1. Basic Evaluation (`audio_caption_matching.py`)
Simple evaluation script that matches generated captions with ground truth captions using similarity scores.

### 2. Advanced Evaluation (`audio_caption_matching_advanced.py`)
Advanced version with random smoothing support and more detailed metrics.

### 3. Multiple Choice Evaluation (`audio_caption_multiple_choice.py`) ⭐ **Recommended**
Treats the task as a multiple-choice question where the model selects the best caption from available options for each audio.

## Usage

### Prerequisites
```bash
pip install torch transformers librosa pandas numpy scikit-learn tqdm
```

### Basic Usage

```bash
# Run multiple choice evaluation (recommended)
python audio_eval/audio_caption_multiple_choice.py \
    --benchmark-csv audio_eval/CompA_order/CompA_order_benchmark.csv \
    --audio-folder audio_eval/CompA_order/CompA_order_files \
    --model-name Qwen/Qwen2-Audio-7B \
    --max-samples 10  # Start with small number for testing
```

### With Random Smoothing

```bash
# Run with random smoothing for robustness evaluation
python audio_eval/audio_caption_multiple_choice.py \
    --benchmark-csv audio_eval/CompA_order/CompA_order_benchmark.csv \
    --audio-folder audio_eval/CompA_order/CompA_order_files \
    --model-name Qwen/Qwen2-Audio-7B \
    --use-random-smoothing \
    --sigma 0.1 \
    --num-copy 5 \
    --max-samples 10
```

### Using the Shell Script

```bash
# Make the script executable
chmod +x audio_eval/run_evaluation.sh

# Run the evaluation script
./audio_eval/run_evaluation.sh
```

## Command Line Arguments

- `--benchmark-csv`: Path to the benchmark CSV file
- `--audio-folder`: Path to the folder containing audio files
- `--model-name`: HuggingFace model name (default: Qwen/Qwen2-Audio-7B)
- `--device`: Device to use (cuda/cpu)
- `--use-random-smoothing`: Enable random smoothing for robustness evaluation
- `--sigma`: Standard deviation for Gaussian noise (default: 0.1)
- `--num-copy`: Number of noisy copies for random smoothing (default: 5)
- `--max-samples`: Maximum number of samples to evaluate (for testing)
- `--output-file`: Output file to save results

## Evaluation Metrics

The scripts compute the following metrics:

1. **Accuracy**: Percentage of correct caption selections
2. **Precision/Recall/F1**: Classification metrics for the multiple choice task
3. **Average Confidence**: Average similarity score of selected captions
4. **Detailed Results**: Per-sample results with generated captions and predictions

## Output Format

Results are saved as JSON files containing:
- Overall metrics (accuracy, precision, recall, F1)
- Per-sample results with:
  - Audio files and ground truth captions
  - Generated captions by the model
  - Model predictions and confidence scores
  - Correctness of predictions

## Example Output

```
=== Multiple Choice Evaluation Results ===
Overall Accuracy: 0.7500
Average Confidence: 0.8234
Precision: 0.7500
Recall: 0.7500
F1 Score: 0.7500
Correct Predictions: 15/20
```

## Notes

- The multiple choice approach is more aligned with the original benchmark design
- Random smoothing can be used to evaluate model robustness to audio perturbations
- Start with a small number of samples (`--max-samples 10`) for testing before running on the full dataset
- Make sure the audio files are accessible in the specified folder 