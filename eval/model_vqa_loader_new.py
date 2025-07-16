import argparse
import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, RobertaTokenizer, RobertaModel
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import _apply_random_smoothing, get_roberta_embeddings, make_clustering
import shortuuid

# Add for saving images
import torchvision.utils as vutils

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-model-name', type=str, default='llava-hf/llava-1.5-7b-hf', help='HuggingFace model name')
    parser.add_argument('--image-folder', type=str, required=True)
    parser.add_argument('--question-file', type=str, required=True)
    parser.add_argument('--answers-file', type=str, default='answer.jsonl')
    parser.add_argument('--sigma', type=float, default=0.1, help='Standard deviation for Gaussian noise (randomized smoothing)')
    parser.add_argument('--num-copy', type=int, default=10, help='Number of noisy copies to generate per image')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    from transformers import LlavaForConditionalGeneration
    model = LlavaForConditionalGeneration.from_pretrained(
        args.hf_model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.hf_model_name, use_fast=False, trust_remote_code=True)
    model.eval()

    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
    roberta_model.eval()

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, 'w')

    # Directory to save noisy images
    noisy_dir = 'noisy_samples'
    os.makedirs(noisy_dir, exist_ok=True)

    with open(args.question_file, 'r') as f:
        questions = [json.loads(line) for line in f]

    for q_idx, line in enumerate(tqdm(questions)):
        image_path = os.path.join(args.image_folder, line['image'])
        question = line['text']
        idx = line['question_id']
        image = load_image(image_path)
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
        noisy_images = _apply_random_smoothing(image_tensor, sigma=args.sigma, num_copy=args.num_copy)
        noisy_images = noisy_images.clamp(0.0, 1.0)

        # Save noisy images for the first five questions
        if q_idx < 20:
            for copy_idx in range(args.num_copy):
                noisy_img = noisy_images[copy_idx]
                # Convert to PIL Image
                np_img = (noisy_img.cpu().numpy() * 255).astype(np.uint8)
                if np_img.shape[0] == 3:  # (C, H, W)
                    np_img = np.transpose(np_img, (1, 2, 0))
                pil_img = Image.fromarray(np_img)
                save_path = os.path.join(noisy_dir, f"q{idx}_copy{copy_idx}.png")
                pil_img.save(save_path)

        all_outputs = []
        all_user_messages = []
        with torch.no_grad():
            for copy_idx in range(args.num_copy):
                noisy_img = noisy_images[copy_idx:copy_idx+1]
                inputs = processor(images=noisy_img, text=prompt, return_tensors='pt').to(device, torch.float16)
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
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": question,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": args.hf_model_name,
            "metadata": {}
        }) + "\n")
    ans_file.close()

if __name__ == "__main__":
    main() 