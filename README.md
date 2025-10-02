# SmoothGuard

Repository for the Paper "SmoothGuard: Defending Multimodal Large Language Models with Noise Perturbation and Clustering Aggregation" (ICDM 2025 Workshop)

A robust defense mechanism for Multimodal Large Language Models (MLLMs) using randomized smoothing. This repository enables robust answer selection for Visual Question Answering (VQA) by adding noise to input images, generating multiple outputs, and clustering results to select the most representative answer. Supports models like Qwen, BLIP-2, and others via HuggingFace Transformers.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Notes](#notes)
- [Citation](#citation)
- [License](#license)
- [Contact / Contributing](#contact--contributing)

## Features

- **Randomized Smoothing**: Adds noise to images for robust inference.
- **Model Support**: Works with Qwen, BLIP-2, and other HuggingFace models.
- **Clustering-based Answer Selection**: Selects robust answers from multiple noisy outputs.
- **Flexible Evaluation**: Tools for VQA evaluation and noisy image inspection.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GuangzhiSu/SmoothGuard.git
cd SmoothGuard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
SmoothGuard/
├── model-safetybench-photo.py           # Defense evaluation with SmoothGuard
├── model-safety-bench-normal-photo.py   # Baseline evaluation without defense
├── evaluation.py                         # ASR metric computation
├── utils.py                              # Utility functions (smoothing, clustering)
├── requirements.txt                      # Python dependencies
├── run_all_photo.sh                      # Batch script for sigma sweep experiments
├── run_formol_photo.sh                   # Batch script for baseline evaluation
├── MM-SafetyBench/                       # Dataset directory
│   ├── processed_questions/              # Safety benchmark questions
│   └── imgs/                             # Associated images
├── eval/                                 # Additional evaluation benchmarks
│   ├── pope/                             # POPE benchmark and scripts
│   └── bench-in-the-wild/               # Bench-in-the-wild benchmark and scripts
├── .gitignore
├── README.md
└── LICENSE
```

## Usage

### Preparing the Dataset

1. Download and prepare the MM-SafetyBench dataset(https://github.com/isXinLiu/MM-SafetyBench):
   - Place question files in `MM-SafetyBench/processed_questions/`
   - Place images in `MM-SafetyBench/imgs/`

2. Prepare the adversarial image (e.g., universal adversarial perturbation(https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models)):
   - Example: `prompt_constrained_32.bmp`

### Quick Start

#### Run Baseline (No Defense)

```bash
python model-safety-bench-normal-photo.py \
    --hf-model-name <model_path> \
    --question-dir MM-SafetyBench/processed_questions \
    --image-dir MM-SafetyBench/imgs \
    --output-dir results_baseline \
    --universal-image <path_to_adversarial_image.bmp> \
    --device cuda
```

#### Run with SmoothGuard Defense

```bash
python model-safetybench-photo.py \
    --hf-model-name <model_path> \
    --question-dir MM-SafetyBench/processed_questions \
    --image-dir MM-SafetyBench/imgs \
    --output-dir results_defended \
    --universal-image <path_to_adversarial_image.bmp> \
    --sigma 0.10 \
    --num-copy 10 \
    --device cuda \
    --scenario 01-Illegal_Activity
```


## Evaluation

### Adversarial Robustness Evaluation (ASR Testing)

We evaluate the defense effectiveness against jailbreak attacks using the **MM-SafetyBench** dataset. The evaluation consists of two modes:

#### 1. Baseline Evaluation (No Defense)

Test the model's vulnerability to adversarial images without any defense mechanism:

```bash
python model-safety-bench-normal-photo.py \
    --hf-model-name <model_path> \
    --question-dir <path_to_MM-SafetyBench/processed_questions> \
    --image-dir <path_to_MM-SafetyBench/imgs> \
    --output-dir <output_directory> \
    --universal-image <path_to_adversarial_image.bmp> \
    --device cuda \
    --max-new-tokens 128 \
    --variant SD
```

**Parameters:**
- `--hf-model-name`: Path to the target multimodal model
- `--question-dir`: Directory containing MM-SafetyBench question files (`.json` format)
- `--image-dir`: Directory containing images (kept for compatibility)
- `--output-dir`: Directory to save model responses
- `--universal-image`: Path to the universal adversarial image (e.g., `prompt_constrained_32.bmp`)
- `--variant`: Question variant type (`SD`, `SD_TYPO`, or `TYPO`)

#### 2. Defense Evaluation (With SmoothGuard)

Test the model with SmoothGuard defense mechanism enabled:

```bash
python model-safetybench-photo.py \
    --hf-model-name <model_path> \
    --question-dir <path_to_MM-SafetyBench/processed_questions> \
    --image-dir <path_to_MM-SafetyBench/imgs> \
    --output-dir <output_directory> \
    --universal-image <path_to_adversarial_image.bmp> \
    --sigma 0.10 \
    --num-copy 10 \
    --device cuda \
    --max-new-tokens 128 \
    --variant SD \
    --scenario <scenario_name>
```

**Defense Parameters:**
- `--sigma`: Noise perturbation level (recommended: 0.05-0.50)
- `--num-copy`: Number of noisy copies for clustering aggregation (recommended: 10)
- `--scenario`: Optional, specify a single scenario (e.g., `01-Illegal_Activity`)

#### 3. Computing ASR Metrics

After generating responses, evaluate the Attack Success Rate (ASR):

```bash
python evaluation.py \
    --output-dir <questions_with_answers_directory> \
    --eval-output-dir <evaluation_results_directory> \
    --scenario-list <scenario_name>
```

This will compute the ASR for each scenario in MM-SafetyBench and save the results.



## Citation

If you use this codebase, please cite:

```bibtex
@inproceedings{su2025smoothguard,
  title={SmoothGuard: Defending Multimodal Large Language Models with Noise Perturbation and Clustering Aggregation},
  author={Su, Guangzhi and [Other Authors]},
  booktitle={ICDM 2025 Workshop},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact / Contributing

For questions or contributions, please:
- Open an issue on GitHub
- Submit a pull request
- Contact: [Your contact information]

