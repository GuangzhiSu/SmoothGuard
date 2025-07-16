# MLLM Defense Using Random Smoothing

This project implements a defense mechanism for Multimodal Large Language Models (MLLMs) using randomized smoothing. The core idea is to add noise to input images, generate multiple outputs, and use clustering to select the most robust answer. The codebase supports models like LLaVA and BLIP-2 via HuggingFace Transformers.

## Project Structure

```
MLLM-defense-using-random-smoothing/
├── inference.py
├── experiment.py
├── experiments.sh
├── utils.py
├── models.txt
├── eval/
│   ├── model_vqa_loader_new.py
│   └── ...
├── data/
│   └── ...
├── scripts/
│   └── ...
├── .gitignore
├── __pycache__/
└── ...
```

## Main Files and Folders

### Top-Level Files

- **inference.py**
  - Runs inference on a dataset using a fixed noise level (sigma). Adds noise to images, generates multiple outputs per input, and clusters the results to select the most robust answer.
  - Usage: `python inference.py --hf-model-name <model> --image_file <img> --prompt-file <jsonl> ...`

- **experiment.py**
  - Sweeps over a range of sigma values to empirically determine the best noise level for randomized smoothing. Saves results for each sigma.
  - Usage: `python experiment.py --hf-model-name <model> --image_file <img> --prompt-file <jsonl> ...`

- **experiments.sh**
  - Example bash script to run experiment.py with various arguments. Useful for batch experiments.

- **utils.py**
  - Contains utility functions for reading prompts, loading images, applying random smoothing, extracting embeddings, and clustering outputs.

- **models.txt**
  - Text file listing available or recommended model names for use with the scripts.

### Folders

- **eval/**
  - Contains evaluation scripts and tools.
  - **model_vqa_loader_new.py**: Loads a VQA dataset, applies noise to images, runs inference with LLaVA (via HuggingFace), clusters outputs, and saves the most representative answer. Also saves noisy images for inspection.

- **data/**
  - (Not shown in detail) Presumably contains datasets, images, or prompt files used for experiments.

- **scripts/**
  - (Not shown in detail) May contain additional helper scripts for running or analyzing experiments.

- **__pycache__/**
  - Python bytecode cache directory (auto-generated).

- **.gitignore**
  - Specifies files/folders to be ignored by git version control.

## How to Run

1. **Install dependencies** (see requirements for HuggingFace Transformers, torch, scikit-learn, etc.).
2. **Prepare your data** (images, prompt files, etc.).
3. **Run experiments**:
   - To sweep sigma: `python experiment.py ...`
   - To run inference with a fixed sigma: `python inference.py ...`
   - To evaluate on VQA: `python eval/model_vqa_loader_new.py --image-folder <dir> --question-file <jsonl> ...`
4. **Check outputs**: Results are saved as `.jsonl` files. Noisy images (for inspection) are saved in `noisy_samples/` by `model_vqa_loader_new.py`.

## Notes
- All model loading and inference is done via HuggingFace Transformers (no local LLaVA code required).
- The code supports robust answer selection via clustering of multiple noisy outputs.
- For more details on arguments, see the top of each script or run with `--help`.

---

Feel free to modify or extend this README as your project evolves! 