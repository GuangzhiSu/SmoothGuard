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
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if "llava" in args.hf_model_name:
        from transformers import LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            args.hf_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(device)
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
    elif "blip2" in args.hf_model_name.lower():
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        model = Blip2ForConditionalGeneration.from_pretrained(
            args.hf_model_name, device_map="auto"
        )
        processor = Blip2Processor.from_pretrained(args.hf_model_name)
    else:
        raise ValueError(f"Model {args.hf_model_name} not supported")
    model.eval()

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, 'w')

    with open(args.question_file, 'r') as f:
        questions = [json.loads(line) for line in f]

    for q_idx, line in enumerate(tqdm(questions)):
        image_path = os.path.join(args.image_folder, line['image'])
        question = line['text']
        idx = line['question_id']
        image = load_image(image_path)
        if "qwen" in args.hf_model_name.lower():
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image", "image": image_path},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            image_tensor, video_tensor = process_vision_info(conversation)
        elif "llava" in args.hf_model_name.lower():
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


        all_user_messages = []
        with torch.no_grad():
            if "blip2" in args.hf_model_name.lower():
                inputs = processor(image, question, return_tensors='pt')
            elif "llava" in args.hf_model_name.lower():
                inputs = processor(images=image, text=[prompt], return_tensors='pt').to(device, torch.float16)
            else:
                inputs = processor(images=image_tensor, text=[prompt], return_tensors='pt').to(device, torch.float16)
            input_ids = inputs['input_ids']
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen_ids = output_ids[0][input_ids.shape[-1]:]
            output = processor.decode(gen_ids, skip_special_tokens=True).strip()
            all_user_messages.append(question)

        print("output", output, flush=True)
        print("Input question:", question, flush=True)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": question,
            "text": output,
            "answer_id": ans_id,
            "model_id": args.hf_model_name,
            "metadata": {}
        }) + "\n")
    ans_file.close()

if __name__ == "__main__":
    main() 