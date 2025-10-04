<div align="center">
  
# SmoothGuard: Defending Multimodal Large Language Models with Noise Perturbation and Clustering Aggregation

---

**Guangzhi Su<sup>1,★</sup>, Shuchang Huang<sup>2,★</sup>, Yutong Ke<sup>1</sup>  
Zhuohang Liu<sup>1</sup>, Long Qian<sup>1</sup>, Kaizhu Huang<sup>1</sup>**

<sup>★</sup>Equal Contribution  

<sup>1</sup>Duke Kunshan University  <sup>2</sup>Independent Researcher

**International Conference on Data Mining (ICDM) workshop, 2025**

</div>
<p align="center">
  <img src="Assets/images/framework with white background.png" alt="SmoothGuard Overview" width="800">
</p>

SmoothGuard is a lightweight, model-agnostic defense for multimodal large language models (MLLMs) that enhances robustness against adversarial attacks. It applies randomized smoothing with Gaussian noise and clustering-based aggregation to filter out adversarial responses while preserving utility. Tested on POPE, Bench-in-the-Wild, and MM-SafetyBench, SmoothGuard achieves strong resistance to attacks without retraining or modifying model architecture.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)
- [Contact / Contributing](#contact--contributing)

## Features

- **Randomized Smoothing**: Adds noise to image and audio(to be updated) for robust inference.
- **Model Support**: Works with Qwen, Llava, and other HuggingFace models.
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



## Usage

### Preparing the Dataset

#### SafetyBench

1. Download and prepare the **MM-SafetyBench** dataset from [MM-SafetyBench GitHub](https://github.com/isXinLiu/MM-SafetyBench).  
   - Place question files in `image_eval/safetybench/processed_questions/`  
   - Place image files in `image_eval/safetybench/imgs/`

2. Prepare the adversarial image (e.g., universal adversarial perturbation) from [Visual-Adversarial-Examples-Jailbreak-Large-Language-Models](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models).  
   - Example file: `prompt_constrained_32.bmp`

#### Bench-in-the-Wild

1. Download the **LLaVA-Bench-in-the-Wild** dataset from [Hugging Face](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/tree/main).  

2. Create a folder named `answers` inside `bench_in_the_wild/`, and move the following files into it:  
   - `answers_gpt4.jsonl`  
   - `bard_0718.jsonl`  
   - `bing_chat_0629.jsonl`  
   These files contain the standard answers produced by different LLMs.

#### POPE

1. Download the **COCO** dataset from [POPE GitHub](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco).  
2. Place the downloaded files under `image_eval/pope/coco/`

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

### Utility Testing

We further evaluate the utility of the model—its ability to maintain normal performance on standard multimodal tasks—using the **Bench-in-the-Wild** and **POPE** benchmarks. Both evaluations are conducted under two modes:

1. **Baseline (No Defense)** – Model runs normally without SmoothGuard.
2. **Defense (With SmoothGuard)** – Model runs with noise-based defense applied.



#### 1. Bench-in-the-Wild Evaluation

**Step 1: Generate Model Answers**

Run the following commands to generate answers for both modes. Replace the model path and file locations as needed.

**Baseline (No Defense)**

```bash
python image_eval/model_vqa_loader_normal.py \
    --hf-model-name <model_path> \
    --image-folder <path_to_bench_in_the_wild/images> \
    --question-file <path_to_bench_in_the_wild/questions.jsonl> \
    --answers-file <output_path>/llava-Instruct-normal.jsonl \
    --device cuda \
    --max-new-tokens 256
```

**Defense (With SmoothGuard)**

```bash
python image_eval/model_vqa_loader_new.py \
    --hf-model-name <model_path> \
    --image-folder <path_to_bench_in_the_wild/images> \
    --question-file <path_to_bench_in_the_wild/questions.jsonl> \
    --answers-file <output_path>/llava-Instruct-new.jsonl \
    --sigma 0.1 \
    --num-copy 10 \
    --device cuda \
    --max-new-tokens 256
```

**Parameters:**

* `--hf-model-name`: Path to the evaluated model.
* `--image-folder`: Directory containing evaluation images.
* `--question-file`: File with benchmark questions.
* `--answers-file`: Path to save generated model responses.
* `--sigma`: Noise perturbation level (applied only in defense mode).
* `--num-copy`: Number of noisy copies aggregated for output (defense mode only).
* `--max-new-tokens`: Maximum response length.



**Step 2: Evaluate with LLM-based Review**

After obtaining answers, evaluate the quality using LLM-based automatic grading:

```bash
python image_eval/bench-in-the-wild/eval_wildbench_qwen.py \
    --question <path_to_bench_in_the_wild/questions.jsonl> \
    --context <path_to_bench_in_the_wild/context.jsonl> \
    --rule <path_to_bench_in_the_wild/rule.json> \
    --answer-list \
        <path_to_bench_in_the_wild/answers/answers_gpt4.jsonl> \
        <path_to_bench_in_the_wild/answers/llava-Instruct-<mode>.jsonl> \
    --output <path_to_bench_in_the_wild/reviews/llava-reviews-<mode>.jsonl>
```

**Where `<mode>`** is either `normal` or `new`.

Then summarize the review results:

```bash
python image_eval/bench-in-the-wild/summarize_gpt_review.py \
    -f <path_to_bench_in_the_wild/reviews/llava-reviews-<mode>.jsonl>
```

This produces aggregated utility metrics that quantify performance quality under both modes.



#### 2. POPE Evaluation

The **POPE** benchmark assesses object hallucination and general reasoning fidelity.

**Step 1: Generate Model Answers**

**Baseline (No Defense)**

```bash
python image_eval/model_vqa_loader_normal.py \
    --hf-model-name <model_path> \
    --image-folder <path_to_pope/val2014> \
    --question-file <path_to_pope/llava_pope_test.jsonl> \
    --answers-file <output_path>/llava-Instruct-normal.jsonl \
    --device cuda \
    --max-new-tokens 256
```

**Defense (With SmoothGuard)**

```bash
python image_eval/model_vqa_loader_new.py \
    --hf-model-name <model_path> \
    --image-folder <path_to_pope/val2014> \
    --question-file <path_to_pope/llava_pope_test.jsonl> \
    --answers-file <output_path>/llava-Instruct-new.jsonl \
    --sigma 0.1 \
    --num-copy 10 \
    --device cuda \
    --max-new-tokens 128
```



**Step 2: Compute POPE Metrics**

After generating responses, evaluate with the POPE scoring script:

```bash
python image_eval/pope/eval_pope.py \
    --annotation-dir <path_to_pope/coco> \
    --question-file <path_to_pope/llava_pope_test.jsonl> \
    --result-file <output_path>/llava_on_pope/llava-Instruct-<mode>.jsonl
```

**Parameters:**

* `--annotation-dir`: Directory containing POPE annotation files.
* `--question-file`: File with POPE question data.
* `--result-file`: Model output file to evaluate.



### Summary

* **Normal Mode:** Evaluates baseline model performance on standard benchmarks.
* **Defense Mode (SmoothGuard):** Evaluates the model with robustness techniques applied.

Together, these utility evaluations measure how well the model retains accuracy and reasoning capability while resisting adversarial perturbations.




## Citation

If you use this codebase, please cite:

```bibtex
@inproceedings{su2025smoothguard,
  title={SmoothGuard: Defending Multimodal Large Language Models with Noise Perturbation and Clustering Aggregation},
  author={Guangzhi Su, Shuchang Huang, Yutong Ke, Zhuohang Liu, Long Qian, and Kaizhu Huang},
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
- Contact: ["Guangzhi Su" <guangzhi.su@dukekunshan.edu.cn>]

