import argparse
import json
import os
from PIL import Image
from tqdm import tqdm
import torch

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-model-name', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--question-dir', type=str, required=True, help='e.g. data/processed_questions')
    parser.add_argument('--image-dir', type=str, required=True, help='dummy, kept for backward compatibility')
    parser.add_argument('--output-dir', type=str, required=True, help='e.g. questions_with_answers')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--variant', type=str, default='SD', choices=['SD', 'SD_TYPO', 'TYPO'])
    # parameter
    parser.add_argument('--universal-image', type=str,
                        default='prompt_constrained_32.bmp',
                        help='Path to the universal adversarial image (e.g. prompt_constrained_32.bmp)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # model
    model_path = ""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()

    # get photp
    universal_img_path = args.universal_image
    assert os.path.isfile(universal_img_path), f"Universal image not found: {universal_img_path}"
    universal_image = load_image(universal_img_path)

    os.makedirs(args.output_dir, exist_ok=True)

    for file_name in os.listdir(args.question_dir):
        if not file_name.endswith('.json'):
            continue
        print(f"Processing file: {file_name}")
        question_path = os.path.join(args.question_dir, file_name)
        with open(question_path, 'r') as f:
            questions = json.load(f)

        output_data = {}
        for qid, qinfo in tqdm(questions.items()):
            # if args.variant == 'SD':
            #     question = qinfo.get('Rephrased Question(SD)', "")
            # elif args.variant in ('SD_TYPO', 'TYPO'):
            #     question = qinfo.get('Rephrased Question', "")
            # else:
            #     continue
            # if not question:
            #     continue
            question = qinfo.get('Question', "")
            print("Question:", question, flush=True)
            
            if not question:
                continue

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image", "image": universal_image},  # PIL.Image
                    ],
                }
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(text=prompt, images=universal_image, return_tensors='pt').to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False
                )
                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            qinfo = dict(qinfo)
            qinfo.setdefault('ans', {})
            qinfo['ans'][args.hf_model_name] = {
                "text": response,
                "metadata": {
                    "universal_image": universal_img_path
                }
            }
            output_data[qid] = qinfo

        output_path = os.path.join(args.output_dir, file_name)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()