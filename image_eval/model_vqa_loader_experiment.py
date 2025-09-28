import argparse
import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, RobertaTokenizer, RobertaModel, CLIPImageProcessor, LlamaTokenizer
from transformers.models.llava.processing_llava import LlavaProcessor
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import _apply_random_smoothing, get_roberta_embeddings, make_clustering
import shortuuid
import matplotlib.pyplot as plt
import csv

# Add for saving images
import torchvision.utils as vutils

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-model-name', type=str, default='liuhaotian/llava-v1.5-7b', help='HuggingFace model name')
    parser.add_argument('--image-folder', type=str, required=True)
    parser.add_argument('--question-file', type=str, required=True)
    parser.add_argument('--answers-file', type=str, default='answer.jsonl')
    # Deprecated single sigma; kept for backward compatibility if range not provided
    parser.add_argument('--sigma', type=float, default=None, help='Standard deviation for Gaussian noise (deprecated if sigma range provided)')
    parser.add_argument('--sigma-start', type=float, default=None, help='Start of sigma sweep (inclusive)')
    parser.add_argument('--sigma-end', type=float, default=None, help='End of sigma sweep (inclusive)')
    parser.add_argument('--sigma-step', type=float, default=0.05, help='Step size for sigma sweep')
    parser.add_argument('--num-copy', type=int, default=10, help='Number of noisy copies to generate per image')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--subset-size', type=int, default=None, help='Number of questions to use (random subset)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if "llava" in args.hf_model_name:
        from transformers import LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            args.hf_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        # vision_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # tokenizer = LlamaTokenizer.from_pretrained(args.hf_model_name, use_fast=False)
        processor = LlavaProcessor.from_pretrained(args.hf_model_name)
    elif "qwen" in args.hf_model_name.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from qwen_vl_utils import process_vision_info
        import accelerate

        # default: Load the model on the available device(s)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.hf_model_name, torch_dtype="auto", device_map="auto"
        )

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    else:
        raise ValueError(f"Model {args.hf_model_name} not supported")
    model.eval()

    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
    roberta_model.eval()

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, 'w')

    # Prepare sigma sweep values
    sigma_values = []
    if args.sigma_start is not None and args.sigma_end is not None:
        current_sigma = args.sigma_start
        # Protect against floating point accumulation by rounding to 3 decimals
        while current_sigma <= args.sigma_end + 1e-9:
            sigma_values.append(round(current_sigma, 3))
            current_sigma = round(current_sigma + args.sigma_step, 10)
    else:
        # Fallback to single sigma if provided, else default 0.1
        fallback_sigma = 0.1 if args.sigma is None else args.sigma
        sigma_values = [float(fallback_sigma)]

    # Directory to save noisy images
    noisy_dir = 'noisy_samples'
    os.makedirs(noisy_dir, exist_ok=True)

    with open(args.question_file, 'r') as f:
        questions = [json.loads(line) for line in f]
    
    # Apply subset if specified
    if args.subset_size and args.subset_size < len(questions):
        import random
        random.seed(42)  # Fixed seed for reproducibility
        questions = random.sample(questions, args.subset_size)
        print(f"Using subset of {len(questions)} questions out of {len(questions) + (len(questions) - args.subset_size)} total")

    # Collect per-sigma stability metrics
    sigma_to_stabilities = {}

    for sigma_value in sigma_values:
        per_sigma_stabilities = []
        for q_idx, line in enumerate(tqdm(questions, desc=f"sigma={sigma_value:.3f}")):
            image_path = os.path.join(args.image_folder, line['image'])
            question = line['text']
            idx = line['question_id']
            image = load_image(image_path)
            if "qwen" in args.hf_model_name.lower():
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
                # Preparation for inference following Qwen template
                text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(conversation)
                # Get the base inputs for Qwen
                base_inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device, torch.float16)
                
                # For random smoothing, we need to work with the pixel_values from base_inputs
                image_tensor = base_inputs['pixel_values']
        
            else:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image"},
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                image_tensor = processor.image_processor(images=image, return_tensors='pt')['pixel_values']

            # Add noise
            noisy_images = _apply_random_smoothing(image_tensor, sigma=sigma_value, num_copy=args.num_copy)

            all_outputs = []
            all_user_messages = []
            with torch.no_grad():
                for copy_idx in range(args.num_copy+1):
                    noisy_img = noisy_images[copy_idx:copy_idx+1]
                    if "qwen" in args.hf_model_name.lower():
                        # For Qwen, use base_inputs and replace pixel_values with noisy tensor
                        inputs = dict(base_inputs)
                        inputs['pixel_values'] = noisy_img.to(device, torch.float16)
                    else:
                        inputs = processor(images=noisy_img, text=[prompt], return_tensors='pt').to(device, torch.float16)
                    input_ids = inputs['input_ids']
                    output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                    gen_ids = output_ids[0][input_ids.shape[-1]:]
                    output = processor.decode(gen_ids, skip_special_tokens=True)
                    all_outputs.append(output)
                    all_user_messages.append(question)

            print("all_outputs", all_outputs, flush=True)
            # Cluster the outputs and select the most representative
            all_embeddings = [get_roberta_embeddings(out, roberta_tokenizer, roberta_model) for out in all_outputs]
            all_embeddings = np.vstack(all_embeddings)
            clustering_result = make_clustering(all_embeddings, all_outputs, all_user_messages)
            outputs = clustering_result['continuation']

            # Compute a stability metric: majority cluster ratio
            from sklearn.cluster import KMeans
            kmeans_tmp = KMeans(n_clusters=2).fit(all_embeddings)
            labels_tmp = kmeans_tmp.labels_
            majority = max(np.sum(labels_tmp == 0), np.sum(labels_tmp == 1))
            stability_ratio = float(majority) / float(len(labels_tmp))
            per_sigma_stabilities.append(stability_ratio)

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": question,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": args.hf_model_name,
                "metadata": {"sigma": sigma_value, "stability_ratio": stability_ratio}
            }) + "\n")

        sigma_to_stabilities[sigma_value] = per_sigma_stabilities
    ans_file.close()

    # Aggregate metrics and save CSV and plot
    results_dir = os.path.dirname(args.answers_file) or '.'
    csv_path = os.path.join(results_dir, 'sigma_sweep_results.csv')
    png_path = os.path.join(results_dir, 'sigma_sweep_results.png')

    rows = [(sigma, float(np.mean(vals)), float(np.std(vals))) for sigma, vals in sigma_to_stabilities.items()]
    # Sort by sigma
    rows.sort(key=lambda x: x[0])
    with open(csv_path, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["sigma", "avg_stability", "std_stability"])
        writer.writerows(rows)

    sigmas_plot = [r[0] for r in rows]
    means_plot = [r[1] for r in rows]
    stds_plot = [r[2] for r in rows]

    plt.figure(figsize=(7, 4))
    plt.errorbar(sigmas_plot, means_plot, yerr=stds_plot, fmt='-o', capsize=3)
    plt.xlabel('Sigma')
    plt.ylabel('Stability (majority cluster ratio)')
    plt.title('Effect of Sigma on Output Stability')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(png_path)
    print(f"Saved CSV to {csv_path} and plot to {png_path}")

if __name__ == "__main__":
    main() 