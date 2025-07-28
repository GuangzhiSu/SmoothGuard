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
    parser.add_argument('--image-dir', type=str, required=True, help='e.g. data/imgs')
    parser.add_argument('--output-dir', type=str, required=True, help='e.g. questions_with_answers')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--variant', type=str, default='SD', choices=['SD', 'SD_TYPO', 'TYPO'])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     args.hf_model_name,
    #     torch_dtype="auto",
    #     device_map="auto"
    # )
    # processor = AutoProcessor.from_pretrained(args.hf_model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/data1/yutong/MLLM-defense-using-random-smoothing/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("/data1/yutong/MLLM-defense-using-random-smoothing/Qwen2.5-VL-7B-Instruct")
    model.eval()

    def process_vision_info(conversation):
        # conversation: [{"role": "user", "content": [{"type": "text", ...}, {"type": "image", ...}]}]
        # return (image_tensor, video_tensor)
        for item in conversation[0]['content']:
            if item['type'] == 'image':
                image_path = item['image']
                if isinstance(image_path, str):
                    img = load_image(image_path)
                else:
                    img = image_path
                image_tensor = processor.image_processor(img, return_tensors='pt')['pixel_values']
                return image_tensor, None
        return None, None

    os.makedirs(args.output_dir, exist_ok=True)

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
            image = load_image(image_path)  # 这里是 PIL.Image
            inputs = processor(text=prompt, images=image, return_tensors='pt').to(device)
            with torch.no_grad():
                inputs = processor(text=prompt, images=image, return_tensors='pt').to(device)
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False
                )
                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            # Store the response
            qinfo = dict(qinfo)  
            qinfo.setdefault('ans', {})
            qinfo['ans'][args.hf_model_name] = {"text": response}
            output_data[qid] = qinfo

        output_path = os.path.join(args.output_dir, file_name)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()