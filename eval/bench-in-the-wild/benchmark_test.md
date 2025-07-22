# llava-bench-in-the-wild: 

## Basic Info
- Important columns: question, image, caption, gpt_answer, category
- Domain: Open-domain, real-world images (e.g., photographs, screenshots, memes, infographics, artworks).
- Tasks: Open-ended visual question answering, captioning, and reasoning.
- Evaluation Style: Largely qualitative, but with GPT-based or human evaluations used to rate the model's output for correctness, helpfulness, and reasoning quality.
- Question Types: Include commonsense reasoning, fine-grained recognition, OCR, spatial understanding, multi-turn conversation, etc.
- Motivation: To assess how well multimodal models generalize beyond curated academic datasets into "in-the-wild" settings—mirroring user-level tasks.



## How to merge the benchmark with Smoothguard
- make a new folder bench_in_the_wild(or any name you like)
- Extract contents of [llava-bench-in-the-wild](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/tree/main) to the folder.
- make a folder inside bench_in_the_wild called answers, and move "answers_gpt4.jsonl","bard_0718.jsonl","bing_chat_0629" into the folder. These are the 'standard answers' produced by different LLMs(used in later steps).

## How to run 

- load_model run (...sh:the same as pope)
- eval run eval_wildbench_qwen.py(equivalent to eval_pope.py for pope)
- run (summarize_gpt_reviews.py(copy from llava/eval ))

scripts/v1_5/eval/llavabench.sh

/playground/data/eval/llava-bench-in-the-wild

## Something else 

- Each image is usually associated with more than one question (with same caption though). If we only consider adding noise to images, we might be able to 1. reduce the benchmark test computation load 2. Compare: for the same image and same randomized smoothing on image, if the result of different questions have similar trend of changes.


## Original sh file to run the whole process
(only need to change the structure of the first python command to the same one as pope, and change the file path of other commands)
```
python -m llava.eval.model_vqa \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-13b.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-13b.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-13b.jsonl
```