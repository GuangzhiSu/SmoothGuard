import argparse
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    

    
    num_copy = 5
    
    # perturbed_images = []
    
    images_tensor = process_images(
        images,
        image_processor,
        model.config,
        sigma=0.50,              # Gaussian noise for randomized smoothing
        noise_enabled=True,       # Enable noise
        num_copy=num_copy     # Generate multiple disturbed images
    ).to(model.device, dtype=torch.float16)
    
    # perturbed_images.append(images_tensor)
    # print('images_tensor',images_tensor.shape) #torch.Size([5, 3, 336, 336])

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    
    # print('input_ids.shape',input_ids.shape)
    # print('input_ids',input_ids)
    repeated_input_ids = input_ids.repeat_interleave(num_copy, dim=0)
    # print('repeated_input_ids',repeated_input_ids)
    # print("repeated_input_ids shape before feeding into model:", repeated_input_ids.shape)
    
    outputs = []
    
    # Loop through each slice (num_copy slices)
    for selected_idx in range(num_copy):
        # Select one slice of input_ids and images_tensor
        selected_input_ids = repeated_input_ids[selected_idx:selected_idx+1]  # Select one text input
        selected_images_tensor = images_tensor[selected_idx:selected_idx+1]   # Select one image tensor
        
        assert selected_idx < repeated_input_ids.size(0), "Index out of bounds for input_ids"
        assert selected_idx < images_tensor.size(0), "Index out of bounds for images_tensor"
        
        # # Print shapes (optional)
        # print(f'Selected input_ids shape for slice {selected_idx}:', selected_input_ids.shape)
        # print(f'Selected images_tensor shape for slice {selected_idx}:', selected_images_tensor)

        with torch.inference_mode():
            # print('inputs.shape',repeated_input_ids.shape())
            # print(f"images_tensor shape: {images_tensor.shape}")
            # print(f"position_ids shape: {position_ids.shape}")
            # print(f"attention_mask shape: {attention_mask.shape}")
            
            output_ids = model.generate(
                inputs = selected_input_ids,
                images=selected_images_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
            
            # print('output_ids',output_ids)
            outputs.append(output_ids)



    # Step 3: Load SBERT for semantic embeddings
    semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

   # Store all embeddings and decoded outputs
    all_embeddings = []
    all_decoded_outputs = []

    # Step 4: Process each batch of outputs one by one
    for selected_idx in range(num_copy):
        # Decode the output IDs to text for each subtensor in the batch
        decoded_outputs = tokenizer.batch_decode(outputs[selected_idx], skip_special_tokens=True)

        
        print('decoded_outputs',decoded_outputs) #list
        
        # Step 5: Get embeddings for decoded text using SBERT
        embeddings = semantic_model.encode(decoded_outputs)
        
        # print('embeddings',embeddings) #list

        # Store embeddings and corresponding decoded outputs for clustering later
        all_embeddings.append(embeddings)
        all_decoded_outputs.append(decoded_outputs)  # Assuming only 1 output per loop

    # Convert lists to arrays for clustering
    all_embeddings = np.vstack(all_embeddings)

    # Step 6: Apply K-Means to cluster based on semantic similarity across all samples
    num_clusters = 2  # Adjust this based on your dataset
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(all_embeddings)

    # Step 7: Majority voting for each cluster
    clustered_outputs = {}
    for cluster_id in set(clusters):
        # Gather outputs that belong to the same cluster
        cluster_texts = [all_decoded_outputs[j][0] for j in range(len(all_decoded_outputs)) if clusters[j] == cluster_id]
        
        # Perform majority voting within the cluster
        most_common_output = Counter(cluster_texts).most_common(1)[0][0]
        clustered_outputs[cluster_id] = most_common_output

    # Step 8: Add the final majority vote result to final_outputs list
    # You can either select the majority output for each cluster or for the entire batch
    final_output = [clustered_outputs[cluster_id] for cluster_id in clusters]

    print(final_output)  # Contains the aggregated result for all samples



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
