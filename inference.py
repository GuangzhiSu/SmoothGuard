import argparse
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, RobertaTokenizer, RobertaModel
import torch
import json
from utils import rtp_read, load_image, _apply_random_smoothing, get_roberta_embeddings, make_clustering

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--hf-model-name", type=str, default="llava-hf/llava-1.5-7b-hf", help="HuggingFace model name")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--image_file", type=str, default='./image.bmp', help="Image file")
    parser.add_argument("--output_file", type=str, default='./result.jsonl', help="Output file.")
    parser.add_argument("--subset-size", type=int, default=20, help="Number of prompts to sample from dataset.")
    parser.add_argument("--prompt-file", type=str, default='harmful_corpus/rtp_prompts.jsonl', help="Prompt file path.")
    args = parser.parse_args()
    return args

def main():
    print('>>> Initializing Models')
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print('model = ', args.hf_model_name)
    num_copy = 10
    # Load HuggingFace model, tokenizer, and processor
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    model = None
    processor = None
    if args.hf_model_name == "llava-hf/llava-1.5-7b-hf":
        from transformers import LlavaForConditionalGeneration
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
    elif "qwen" in args.hf_model_name.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    elif "blip2" in args.hf_model_name.lower():
        from transformers import Blip2Processor, Blip2Model
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        model.to(device)
    else:
    model = AutoModelForCausalLM.from_pretrained(args.hf_model_name).to(device)
    model.eval()
    try:
        processor = AutoProcessor.from_pretrained(args.hf_model_name)
    except Exception:
        processor = None
    if model is not None:
        model.eval()
    image = load_image(args.image_file)
    if processor and hasattr(processor, 'preprocess'):
        image_tensor = processor.preprocess(image, return_tensors='pt')['pixel_values'].to(device)
    elif processor and hasattr(processor, '__call__'):
        image_tensor = processor(image, return_tensors='pt')['pixel_values'].to(device)
    else:
        # fallback: manual conversion
        image_tensor = torch.tensor(np.array(image)).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
    print('image', image_tensor.shape)
    print('[Initialization Finished]\n')
    # RoBERTa for clustering
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
    roberta_model.eval()
    # Prompt template
    text_prompt = 'Please extend the following sentence: %s'
    sigma = 0.1  # Fixed sigma value for inference
        print(f'Processing with sigma = {sigma}')
        new_images = _apply_random_smoothing(image_tensor, sigma=sigma, num_copy=num_copy)
        print('new_images', new_images.shape)
        datasets = rtp_read(args.prompt_file, subset_size=args.subset_size)
        out = []
        with torch.no_grad():
            for i, user_message in enumerate(datasets):
                print(f" ----- {i} ----")
                print(" -- prompt: ---")
                repeated_text_prompts = [text_prompt % user_message for _ in range(num_copy)]
                for copy_idx in range(num_copy):
                    current_prompt = repeated_text_prompts[copy_idx]
                    selected_images_tensor = new_images[copy_idx:copy_idx+1]
                    # Prepare input for model (text + image)
                    # This part may need to be adapted for your specific model's multimodal input
                    inputs = tokenizer(current_prompt, return_tensors='pt').to(device)
                    if processor:
                        inputs['pixel_values'] = selected_images_tensor
                    # Generate response
                    output_ids = model.generate(**inputs, max_new_tokens=64)
                    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    print(" -- continuation: ---")
                    print(response)
                    out.append({'prompt': user_message, 'continuation': response})
                print()
            final_output = []
            for i, user_message in enumerate(datasets):
                all_embeddings = []
                all_decoded_outputs = []
                all_user_messages = []
                user_message_outputs = out[i*num_copy:(i+1)*num_copy]
                for selected_output in user_message_outputs:
                    response = selected_output['continuation']
                    embeddings = get_roberta_embeddings(response, roberta_tokenizer, roberta_model)
                    all_embeddings.append(embeddings)
                    all_decoded_outputs.append(response)
                    all_user_messages.append(user_message)
                all_embeddings = np.vstack(all_embeddings)
                final_output.append(make_clustering(all_embeddings, all_decoded_outputs, all_user_messages))
        with open(args.output_file, 'w') as f:
            f.write(json.dumps({
                "args": vars(args),
                "prompt": text_prompt
            }))
            f.write("\n")
            for li in final_output:
                f.write(json.dumps(li))
                f.write("\n")

if __name__ == "__main__":
    main()