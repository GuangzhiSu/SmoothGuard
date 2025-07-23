# MLLM Defense Using Random Smoothing

A robust defense mechanism for Multimodal Large Language Models (MLLMs) using randomized smoothing. This repository enables robust answer selection for Visual Question Answering (VQA) by adding noise to input images, generating multiple outputs, and clustering results to select the most representative answer. Supports models like Qwen, BLIP-2, and others via HuggingFace Transformers.

---

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

---

## Features

- **Randomized Smoothing**: Adds noise to images for robust inference.
- **Model Support**: Works with Qwen, BLIP-2, and other HuggingFace models.
- **Clustering-based Answer Selection**: Selects robust answers from multiple noisy outputs.
- **Flexible Evaluation**: Tools for VQA evaluation and noisy image inspection.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd MLLM-defense-using-random-smoothing
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure

```
MLLM-defense-using-random-smoothing/
├── experiment.py                  # Sigma sweep experiments
├── experiments.sh                 # Example batch script for experiments
├── inference.py                   # Inference with random smoothing
├── models.txt                     # List of supported models
├── utils.py                       # Utility functions
├── requirements.txt               # Python dependencies
├── eval/
│   ├── model_vqa_loader_new.py    # VQA evaluation with smoothing
│   ├── model_vqa_loader_normal.py # VQA evaluation (baseline)
│   ├── pope/                      # POPE benchmark and scripts
│   └── bench-in-the-wild/         # Bench-in-the-wild benchmark and scripts
├── llava_provided/
│   ├── scripts/                   # LLaVA-related scripts (finetuning, conversion, etc.)
│   └── data/                      # Example datasets for various benchmarks
├── .gitignore
├── README.md
└── ...
```

---

## Usage

### 1. Prepare Data

- Place your images and prompt files (e.g., `.jsonl`) in the appropriate data folders (see `llava_provided/data/` for examples).

### 2. Run Inference

- Run inference with a fixed noise level (sigma):
  ```bash
  python inference.py --hf-model-name <model> --image_file <img> --prompt-file <jsonl> --sigma <value> --num-samples <N>
  ```

### 3. Run Experiments

- Sweep over sigma values to find the best noise level:
  ```bash
  python experiment.py --hf-model-name <model> --image_file <img> --prompt-file <jsonl> --sigma-list <list>
  ```
- Example batch run:
  ```bash
  bash experiments.sh
  ```

### 4. Evaluate on VQA Benchmarks

- **POPE Benchmark:**
  - Run the loader:
    ```bash
    python eval/model_vqa_loader_normal.py --hf-model-name <model> --image-folder <dir> --question-file <jsonl> --answers-file <out.jsonl>
    ```
  - Evaluate results:
    ```bash
    python eval/pope/eval_pope.py --annotation-dir <coco_dir> --question-file <jsonl> --result-file <out.jsonl>
    ```

- **Bench-in-the-Wild:**
  - Run the loader:
    ```bash
    python eval/bench-in-the-wild/eval_wildbench_qwen.py
    ```
  - Summarize results:
    ```bash
    python eval/bench-in-the-wild/summarize_gpt_review.py -f <review_file.jsonl>
    ```

- Example answer and review files can be found in `eval/pope/answers/` and `eval/bench-in-the-wild/answers/` and `eval/bench-in-the-wild/reviews/`.

---

## Evaluation

- **Robustness**: Evaluates model robustness by clustering outputs from multiple noisy samples and selecting the most representative answer.
- **Metrics**: Outputs are in `.jsonl` format for easy analysis. Noisy images are saved for qualitative inspection.
- **Benchmarks**: Includes scripts and data for POPE and Bench-in-the-Wild evaluations.

---

## Notes

- All model loading and inference is done via HuggingFace Transformers.
- For Qwen-VL, ensure `qwen_vl_utils.py` is available in your PYTHONPATH or install from the appropriate source.
- For LLaVA, install the official or custom LLaVA package as needed (not on PyPI as of writing).
- For detailed script arguments, run any script with `--help`.

---

## Citation

If you use this codebase, please cite:

```
[Add your citation here]
```

---

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

---

## Contact / Contributing

For questions or contributions, please open an issue or pull request. 