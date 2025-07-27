# Audio-Caption Matching Benchmark

This directory contains evaluation scripts for the CompA_order audio-caption matching benchmark. The task is to correctly match audio files with their corresponding captions, where the captions contain the same words but in different orders to reflect different audio compositions.

## Benchmark Structure

The benchmark consists of:
- **Audio files**: Different compositions of the same audio elements (e.g., animal sounds + human speech)
- **Captions**: Two or three different captions for each set, containing the same words but in different orders:
  - `pair_caption`: Original composition description
  - `reversed_pair_caption`: Reversed order description  
  - `triplet_caption`: Mixed/interleaved composition description (may be "-" for some rows)

**Note**: Some benchmark rows contain only 2 valid audio-caption pairs (with "-" for the third option), which the scripts handle automatically.

## Task Description

For each set of audio files, the model must:
1. Listen to an audio file (randomly selected from available options)
2. Select the correct caption from multiple choice options (2 or 3 depending on available data)
3. The task is evaluated as a direct multiple-choice question where the model outputs the choice number (1, 2, or 3)

## Evaluation Scripts

### 1. Basic Evaluation (`audio_caption_normal.py`) ⭐ **Recommended for Standard Evaluation**
- **Purpose**: Direct multiple-choice evaluation without noise or clustering
- **Features**:
  - No random noise added to audio
  - Simple word overlap (Jaccard similarity) for evaluation
  - Direct model selection from given options
  - Real-time results logging
  - Random selection of one audio-caption pair per row

### 2. Clustering-Based Evaluation (`audio_caption_new.py`) ⭐ **Recommended for Robustness Testing**
- **Purpose**: Evaluation with random smoothing and response clustering for robustness
- **Features**:
  - Adds Gaussian noise to audio files
  - Generates multiple responses from noisy copies
  - Uses RoBERTa embeddings and K-means clustering to determine final prediction
  - Real-time results logging
  - Random selection of one audio-caption pair per row

## Usage

### Prerequisites
```bash
pip install torch transformers librosa pandas numpy scikit-learn tqdm sentence-transformers
```

### Basic Evaluation (No Noise)

```bash
# Run basic evaluation without random smoothing
python audio_eval/audio_caption_normal.py \
    --benchmark-csv audio_eval/CompA_order/CompA_order_benchmark.csv \
    --audio-folder audio_eval/CompA_order/CompA_order_files \
    --model-name Qwen/Qwen2-Audio-7B \
    --device cuda \
    --max-samples 10  # Start with small number for testing
```

### Clustering-Based Evaluation (With Noise)

```bash
# Run with random smoothing and clustering
python audio_eval/audio_caption_new.py \
    --benchmark-csv audio_eval/CompA_order/CompA_order_benchmark.csv \
    --audio-folder audio_eval/CompA_order/CompA_order_files \
    --model-name Qwen/Qwen2-Audio-7B \
    --device cuda \
    --sigma 0.1 \
    --num-copy 5 \
    --max-samples 10
```

### Using the Shell Script

```bash
# Make the script executable
chmod +x audio_eval/run_comparison.sh

# Run both evaluations for comparison
./audio_eval/run_comparison.sh
```

## Command Line Arguments

### Common Arguments (Both Scripts)
- `--benchmark-csv`: Path to the benchmark CSV file
- `--audio-folder`: Path to the folder containing audio files
- `--model-name`: HuggingFace model name (default: Qwen/Qwen2-Audio-7B)
- `--device`: Device to use (cuda, cuda:0, cuda:1, cpu)
- `--max-samples`: Maximum number of samples to evaluate (for testing)
- `--output-file`: Output file to save results

### Clustering Script Specific Arguments
- `--sigma`: Standard deviation for Gaussian noise (default: 0.1)
- `--num-copy`: Number of noisy copies for random smoothing (default: 5)

## Data Handling

### Partial Data Support
The scripts automatically handle benchmark rows with different numbers of valid options:

**2-Option Row Example:**
```
A doorbell chiming succeeded by melodic tunes.,Melodic tunes succeeded by a chiming doorbell.,-,34.wav,25_rev.wav,-
```
- **Processed**: Only the 2 valid audio-caption pairs are used
- **Multiple Choice**: Options 1 and 2 (no option 3)
- **Random Selection**: Picks between the 2 available options

**3-Option Row Example:**
```
Caption1,Caption2,Caption3,file1.wav,file2.wav,file3.wav
```
- **Processed**: All 3 valid audio-caption pairs are used
- **Multiple Choice**: Options 1, 2, and 3
- **Random Selection**: Picks between the 3 available options

### Skipping Logic
- Rows with fewer than 2 valid options are skipped
- Progress messages show which rows are skipped and why

## Evaluation Metrics

The scripts compute the following metrics:

1. **Accuracy**: Percentage of correct caption selections
2. **Total Questions**: Number of processed questions
3. **Correct Predictions**: Number of correct answers
4. **Real-time Progress**: Results are written to JSON after each question

## Output Format

Results are saved as JSON files containing:
- **Metadata**: Overall statistics (total, correct, accuracy)
- **Questions**: Per-question results with:
  - Row index and question ID
  - Selected audio file and available options
  - Model prediction and confidence
  - Correctness of prediction
  - All available caption options

### Example Output Structure
```json
{
  "metadata": {
    "total": 20,
    "correct": 15,
    "accuracy": 0.75
  },
  "questions": [
    {
      "row_index": 0,
      "question_id": "row_0_random_selection",
      "selected_audio_file": "34.wav",
      "candidate_captions": ["Caption1", "Caption2"],
      "correct_answer_index": 0,
      "model_prediction_index": 0,
      "is_correct": true,
      "confidence": 1.0
    }
  ]
}
```

## Key Features

### Real-time Logging
- Results are written to JSON file after each question
- Allows monitoring progress during long evaluations
- Prevents data loss if script is interrupted

### Random Selection
- For each row, randomly selects one audio-caption pair to evaluate
- Reduces evaluation time while maintaining statistical validity
- Uses deterministic seeding for reproducible results

### Robust Error Handling
- Handles device compatibility issues (CUDA/CPU)
- Filters out unsupported model arguments
- Gracefully handles missing or invalid data

## Notes

- **Start Small**: Use `--max-samples 10` for testing before full evaluation
- **Device Selection**: Use `--device cuda` for GPU acceleration, `--device cpu` for CPU-only
- **File Paths**: Ensure audio files are accessible in the specified folder
- **Model Compatibility**: Tested with Qwen2-Audio-7B, may work with other audio models
- **Memory Usage**: Clustering script uses more memory due to multiple model runs and embeddings 