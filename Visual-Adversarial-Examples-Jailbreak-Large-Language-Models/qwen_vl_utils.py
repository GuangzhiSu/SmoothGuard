import torch
from PIL import Image
import numpy as np

def process_vision_info(conversation):
    """
    Process vision information for Qwen2.5-VL model
    Based on the implementation in model_vqa_loader_normal.py
    """
    image_tensor = None
    video_tensor = None
    
    for message in conversation:
        if message["role"] == "user":
            for content in message["content"]:
                if content["type"] == "image":
                    image_path = content["image"]
                    image = Image.open(image_path).convert('RGB')
                    # Convert to tensor and normalize
                    image_array = np.array(image)
                    image_tensor = torch.from_numpy(image_array).float() / 255.0
                    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # CHW format
                    break
    
    return image_tensor, video_tensor
