import argparse
import os
import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import json


def rtp_read(text_file):
    dataset = []
    lines = open(text_file).read().split("\n")
    for li in lines:
        obj = json.loads(li)
        if obj['challenging']:
            dataset.append(obj['prompt']['text'])
    return dataset

def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-path", type=str, default="ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")


    parser.add_argument("--image_file", type=str, default='./image.bmp',
                        help="Image file")
    parser.add_argument("--output_file", type=str, default='./result.jsonl',
                        help="Output file.")

    args = parser.parse_args()
    return args


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image



# ========================================
#             Model Initialization
# ========================================

print('>>> Initializing Models')

from llava_llama_2.utils import get_model
args = parse_args()

print('model = ', args.model_path)


num_copy = 5

def _apply_random_smoothing(images, sigma, num_copy):
    """
    Apply Gaussian noise (randomized smoothing) to the input images.

    :param images: Tensor of input images [batch_size, channels, width, height]
    :param sigma: Standard deviation of the Gaussian noise
    :param num_copy: Number of noisy copies to generate per image
    :return: Images with added Gaussian noise
    """
    noisy_images = []

    # Iterate through each image in the batch
    for image in images:
        # Create `num_copy` noisy versions of the current image
        for _ in range(num_copy):
            noise = torch.randn_like(image) * sigma
            noisy_image = image + noise
            noisy_images.append(noisy_image)
    # print('noisy_images',torch.stack(noisy_images, dim=0))
    # Stack all the noisy images back together
    return torch.stack(noisy_images, dim=0)


tokenizer, model, image_processor, model_name = get_model(args)
model.eval()

image = load_image(args.image_file)


image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
print('image',image.shape)

new_images = _apply_random_smoothing(image, sigma = 0.5, num_copy = 5)
print('new_images',new_images.shape)

print('[Initialization Finished]\n')




from llava_llama_2_utils import prompt_wrapper, generator


my_generator = generator.Generator(model=model, tokenizer=tokenizer)


# ========================================
#             Inference
# ========================================

##  TODO: expose interface.
datasets = rtp_read('harmful_corpus/rtp_prompts.jsonl')




#%s is a placeholder for a user message.
text_prompt = 'Please extend the following sentence: %s'

out = []
with torch.no_grad():

    for i, user_message in enumerate(datasets):

        print(f" ----- {i} ----")
        print(" -- prompt: ---")

        # print(text_prompt % user_message)
        
        # Create a repeated version of the text prompt for num_copy
        repeated_text_prompts = [text_prompt % user_message for _ in range(num_copy)]

        # Process each repeated prompt
        for copy_idx in range(num_copy):
            
            # Use the repeated prompt for this iteration
            current_prompt = repeated_text_prompts[copy_idx]
            selected_images_tensor = new_images[copy_idx:copy_idx+1]   # Select one image tensor
            print('selected_images_tensor',selected_images_tensor.shape)

            # Prepare the text prompt template and the prompt object
            text_prompt_template = prompt_wrapper.prepare_text_prompt(current_prompt)
            prompt = prompt_wrapper.Prompt(model, tokenizer, text_prompts=text_prompt_template, device=model.device)

            # Generate the response for the current repeated prompt
            response = my_generator.generate(prompt, selected_images_tensor)

            print(" -- continuation: ---")
            print(response)
            out.append({'prompt': user_message, 'continuation': response})

        print()

        
    # Step 3: Load SBERT for semantic embeddings
    semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

   # Store all embeddings and decoded outputs
    all_embeddings = []
    all_decoded_outputs = []
    all_user_messages = []  # New list to store the user_message for each output

    # Iterate through all `out` entries (which are continuations for each `user_message`)
    for i, user_message in enumerate(datasets):
        # Extract all continuations for the current user message
        user_message_outputs = out[i*num_copy:(i+1)*num_copy]  # Extracts the responses corresponding to one user_message

        # Process each continuation for this user_message
        for selected_output in user_message_outputs:
            
            # Get the response (continuation) from the dictionary
            response = selected_output['continuation']
            
            # Step 5: Get embeddings for decoded text using SBERT
            embeddings = semantic_model.encode(response)
            
            # print('embeddings',embeddings) #list

            # Store embeddings and corresponding decoded outputs for clustering later
            all_embeddings.append(embeddings)
            all_decoded_outputs.append(response)  # Assuming only 1 output per loop
            all_user_messages.append(user_message)  # Keep track of the original user message

    # Convert lists to arrays for clustering
    all_embeddings = np.vstack(all_embeddings)

    # Step 6: Apply K-Means to cluster based on semantic similarity across all samples
    num_clusters = 2  # Adjust this based on your dataset
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(all_embeddings)

    # Step 7: Find the larger cluster based on number of elements
    final_output = []
    cluster_sizes = {}
    cluster_embeddings = {}
    
    # Collect embeddings and sizes for each cluster
    for cluster_id in set(clusters):
        # Gather outputs that belong to the same cluster
        cluster_texts = [all_decoded_outputs[j] for j in range(len(all_decoded_outputs)) if clusters[j] == cluster_id]
        
        # Get embeddings for the cluster
        cluster_embeddings[cluster_id] = np.array([all_embeddings[j] for j in range(len(all_embeddings)) if clusters[j] == cluster_id])
        
        # Store the size of each cluster
        cluster_sizes[cluster_id] = len(cluster_texts)

    # Step 8: Select the larger cluster
    larger_cluster_id = max(cluster_sizes, key=cluster_sizes.get)

    # Step 9: Compute the centroid of the larger cluster
    larger_cluster_embeddings = cluster_embeddings[larger_cluster_id]
    centroid_embedding = np.mean(larger_cluster_embeddings, axis=0)

    # Step 10: Find the response closest to the centroid
    similarities = cosine_similarity([centroid_embedding], larger_cluster_embeddings)
    closest_idx = np.argmax(similarities)
    
    # Step 11: Select the most representative response from the larger cluster
    larger_cluster_texts = [all_decoded_outputs[j] for j in range(len(all_decoded_outputs)) if clusters[j] == larger_cluster_id]
    most_representative_text = larger_cluster_texts[closest_idx][0]  # Assuming response is in a nested list structure

    # Also keep track of the corresponding user message
    corresponding_user_message = all_user_messages[closest_idx]
    
    # Step 12: Add the selected prompt and response to the final_output list
    final_output.append({
        'prompt': corresponding_user_message, 
        'continuation': most_representative_text
    })
    
    # Final output: the most representative response from the larger cluster
    print("Final Output:", most_representative_text)




with open(args.output_file, 'w') as f:
    f.write(json.dumps({
        "args": vars(args),
        "prompt": text_prompt
    }))
    f.write("\n")

    for li in final_output:
        f.write(json.dumps(li))
        f.write("\n")