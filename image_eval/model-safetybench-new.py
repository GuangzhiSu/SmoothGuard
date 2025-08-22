import argparse
import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import shortuuid

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import _apply_random_smoothing, get_roberta_embeddings, make_clustering

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-model-name', type=str, required=True)
    parser.add_argument('--question-dir', type=str, required=True, help='e.g. data/processed_questions')
    parser.add_argument('--image-dir', type=str, required=True, help='e.g. data/imgs')
    parser.add_argument('--output-dir', type=str, required=True, help='e.g. questions_with_answers')
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--num-copy', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--variant', type=str, default='SD', choices=['SD', 'SD_TYPO', 'TYPO'])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- 加载模型 ----
    if "llava" in args.hf_model_name.lower():
        from transformers import LlavaForConditionalGeneration
        from transformers.models.llava.processing_llava import LlavaProcessor
        model = LlavaForConditionalGeneration.from_pretrained(
            args.hf_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        processor = LlavaProcessor.from_pretrained(args.hf_model_name)
    elif "qwen" in args.hf_model_name.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.hf_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.hf_model_name)
    else:
        raise ValueError(f"Model {args.hf_model_name} not supported")
    model.eval()

    # ---- 加载Roberta做语义聚类 ----
    from transformers import RobertaTokenizer, RobertaModel
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
    roberta_model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    # 遍历每个scenario的question文件
    for file_name in os.listdir(args.question_dir):
        if not file_name.endswith('.json'):
            continue
        print(f"Processing file: {file_name}")
        scenario = file_name.replace('.json', '')
        question_path = os.path.join(args.question_dir, file_name)
        with open(question_path, 'r') as f:
            questions = json.load(f)
        output_data = {}
        for qid, qinfo in tqdm(questions.items()):
            # 选择问题和图片路径
            if args.variant == 'SD':
                image_path = os.path.join(args.image_dir, scenario, 'SD', f'{qid}.jpg')
                question = qinfo.get('Rephrased Question(SD)', "")
            elif args.variant == 'SD_TYPO':
                image_path = os.path.join(args.image_dir, scenario, 'SD_TYPO', f'{qid}.jpg')
                question = qinfo.get('Rephrased Question', "")
            elif args.variant == 'TYPO':
                image_path = os.path.join(args.image_dir, scenario, 'TYPO', f'{qid}.jpg')
                question = qinfo.get('Rephrased Question', "")

            if not os.path.exists(image_path) or not question:
                continue

            image = load_image(image_path)
            all_outputs = []
            all_user_messages = []

            # --- Qwen 和 LLaVA分支 ---
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
                text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                # 直接传 PIL.Image 给 processor
                base_inputs = processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                ).to(device, torch.float16)
                image_tensor = base_inputs['pixel_values']  # (1, C, H, W)
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

            # ---- 随机扰动 ----
            noisy_images = _apply_random_smoothing(image_tensor, sigma=args.sigma, num_copy=args.num_copy)
            # noisy_images: (num_copy+1, C, H, W)

            # 依次推理并聚类
            with torch.no_grad():
                for copy_idx in range(args.num_copy+1):
                    noisy_img = noisy_images[copy_idx:copy_idx+1]
                    if "qwen" in args.hf_model_name.lower():
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

            # 语义聚类
            all_embeddings = [get_roberta_embeddings(out, roberta_tokenizer, roberta_model) for out in all_outputs]
            all_embeddings = np.vstack(all_embeddings)
            clustering_result = make_clustering(all_embeddings, all_outputs, all_user_messages)
            outputs = clustering_result['continuation']

            # ----------- 写入MMSafetyBench兼容格式 -----------
            ans_id = shortuuid.uuid()
            qinfo = dict(qinfo)
            qinfo.setdefault('ans', {})
            qinfo['ans'][args.hf_model_name] = {
                "text": outputs,
                "answer_id": ans_id,
                "model_id": args.hf_model_name,
                "metadata": {
                    "num_noisy_samples": args.num_copy,
                    "random_smoothing_sigma": args.sigma
                }
            }
            output_data[qid] = qinfo

        output_path = os.path.join(args.output_dir, file_name)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()