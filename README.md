# MLLM Defense Using Random Smoothing

A robust defense mechanism for Multimodal Large Language Models (MLLMs) using randomized smoothing. This repository enables robust answer selection for Visual Question Answering (VQA) by adding noise to input images, generating multiple outputs, and clustering results to select the most representative answer. Supports models like LLaVA and BLIP-2 via HuggingFace Transformers.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Evaluation](#evaluation)
- [Notes](#notes)
- [Citation](#citation)
- [License](#license)
- [Contact / Contributing](#contact--contributing)

---

## Features

- **Randomized Smoothing**: Adds noise to images for robust inference.
- **Model Support**: Works with LLaVA, BLIP-2, and other HuggingFace models.
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
   - Required: `transformers`, `torch`, `scikit-learn`, etc.

---

## Usage

### 1. Prepare Data

- Place your images and prompt files (e.g., `.jsonl`) in the `data/` directory.

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

### 4. Evaluate on VQA

- Evaluate using the provided loader:
  ```bash
  python eval/model_vqa_loader_new.py --image-folder <dir> --question-file <jsonl> --output-file <out.jsonl>
  ```

### 5. Check Outputs

- Results are saved as `.jsonl` files.
- Noisy images for inspection are saved in `noisy_samples/` by the evaluation script.

---

## Project Structure

```
MLLM-defense-using-random-smoothing/
├── inference.py                # Inference with random smoothing
├── experiment.py               # Sigma sweep experiments
├── experiments.sh              # Example batch script
├── utils.py                    # Utility functions
├── models.txt                  # List of supported models
├── eval/
│   └── model_vqa_loader_new.py # VQA evaluation script
├── data/                       # Place your data here
├── scripts/                    # Additional scripts
├── noisy_samples/              # Saved noisy images (output)
├── .gitignore
└── ...
```

---

## Evaluation

- **Robustness**: The system evaluates model robustness by clustering outputs from multiple noisy samples and selecting the most representative answer.
- **Metrics**: Outputs are in `.jsonl` format for easy analysis. Noisy images are saved for qualitative inspection.
- **Custom Evaluation**: You can adapt `eval/model_vqa_loader_new.py` for your own datasets or evaluation protocols.

---

## Notes

- All model loading and inference is done via HuggingFace Transformers.
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