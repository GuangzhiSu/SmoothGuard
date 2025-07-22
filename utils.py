import random
import json
import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def rtp_read(text_file, subset_size=20, seed=42):
    dataset = []
    lines = open(text_file).read().split("\n")
    random.seed(seed)
    parsed_objects = []
    for li in lines:
        if not li.strip():
            continue
        obj = json.loads(li)
        if obj['challenging']:
            parsed_objects.append(obj['prompt']['text'])
    if len(parsed_objects) > subset_size:
        dataset = random.sample(parsed_objects, subset_size)
    else:
        dataset = parsed_objects
    return dataset

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def _apply_random_smoothing(images, sigma, num_copy):
    noisy_images = []
    
    # Handle different input formats
    if images.dim() == 2:
        # Qwen format: [height*width, channels] -> treat as single image
        for _ in range(num_copy):
            noise = torch.randn_like(images) * sigma
            noisy_image = images + noise
            noisy_images.append(noisy_image)
        noisy_images.append(images)  # Add original image
        return torch.stack(noisy_images, dim=0)  # [num_images, height*width, channels]
    
    elif images.dim() == 4:
        # Standard format: [batch, channels, height, width]
        for image in images:
            for _ in range(num_copy):
                noise = torch.randn_like(image) * sigma
                noisy_image = image + noise
                noisy_images.append(noisy_image)
            noisy_images.append(image)
        return torch.stack(noisy_images, dim=0)
    else:
        raise ValueError(f"Unsupported image format: {images.dim()}")

def get_roberta_embeddings(text, roberta_tokenizer, roberta_model):
    inputs = roberta_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to('cuda')
    with torch.no_grad():
        outputs = roberta_model(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
    return embeddings

def make_clustering(all_embeddings, all_decoded_outputs, all_user_messages, num_clusters=2):
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(all_embeddings)
    cluster_sizes = {}
    cluster_embeddings = {}
    for cluster_id in set(clusters):
        cluster_embeddings[cluster_id] = np.array([all_embeddings[j] for j in range(len(all_embeddings)) if clusters[j] == cluster_id])
        cluster_sizes[cluster_id] = sum(clusters == cluster_id)
    larger_cluster_id = max(cluster_sizes, key=cluster_sizes.get)
    larger_cluster_embeddings = cluster_embeddings[larger_cluster_id]
    centroid_embedding = np.mean(larger_cluster_embeddings, axis=0)
    similarities = cosine_similarity([centroid_embedding], larger_cluster_embeddings)
    closest_idx = np.argmax(similarities)
    larger_cluster_texts = [all_decoded_outputs[j] for j in range(len(all_decoded_outputs)) if clusters[j] == larger_cluster_id]
    most_representative_text = larger_cluster_texts[closest_idx]
    corresponding_user_message = [all_user_messages[j] for j in range(len(all_user_messages)) if clusters[j] == larger_cluster_id][closest_idx]
    clustering_output = {
        'prompt': corresponding_user_message,
        'continuation': most_representative_text
    }
    print("Final Output:", most_representative_text)
    return clustering_output 