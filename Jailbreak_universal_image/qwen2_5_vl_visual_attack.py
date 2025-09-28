import argparse
import torch
import os
from torchvision.utils import save_image
from PIL import Image
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Visual Attack Demo")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", 
                        help="HuggingFace model name")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=5000, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", default=False, action='store_true')
    parser.add_argument("--save_dir", type=str, default='qwen_output',
                        help="save directory")
    parser.add_argument("--template_img", type=str, default='adversarial_images/clean.jpeg',
                        help="template image path")
    parser.add_argument("--harmful_corpus", type=str, default='harmful_corpus/derogatory_corpus.csv',
                        help="harmful corpus path")

    args = parser.parse_args()
    return args

def load_image(image_path):
    """Load and preprocess image for Qwen2.5-VL"""
    image = Image.open(image_path).convert('RGB')
    return image

def get_model_and_tokenizer(model_name, device):
    """Load Qwen2.5-VL model and tokenizer from HuggingFace"""
    print(f"Loading model: {model_name}")
    
    # Try to import transformers with error handling
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        print("Successfully imported transformers components")
        
        # Try to import qwen_vl_utils for official template approach
        try:
            from qwen_vl_utils import process_vision_info
            print("Successfully imported qwen_vl_utils")
        except ImportError:
            print("Warning: qwen_vl_utils not available, will use fallback approach")
            process_vision_info = None
            
    except Exception as e:
        print(f"Error importing transformers: {e}")
        print("This might be due to PyTorch version compatibility issues.")
        print("Please check your PyTorch and transformers versions.")
        raise e
    
    # Load model - try multiple approaches to avoid torch.compiler issues
    # IMPORTANT: Avoid device_map="auto" for gradient computation
    model = None
    loading_methods = [
        # Method 1: Try without device_map (preferred for gradients)
        lambda: Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", trust_remote_code=True
        ).to(device),
        # Method 2: Try with minimal parameters
        lambda: Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device),
        # Method 3: Try with specific torch_dtype
        lambda: Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        ).to(device),
        # Method 4: Try with low_cpu_mem_usage
        lambda: Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, low_cpu_mem_usage=True, trust_remote_code=True
        ).to(device),
        # Method 5: Try with device_map as last resort (not recommended for gradients)
        lambda: Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
        ),
    ]
    
    for i, method in enumerate(loading_methods, 1):
        try:
            print(f"Trying loading method {i}...")
            model = method()
            print(f"Model loaded successfully with method {i}")
            
            # Check if model is properly on device and can compute gradients
            if hasattr(model, 'device_map') and model.device_map is not None:
                print(f"Warning: Model loaded with device_map: {model.device_map}")
                print("This might interfere with gradient computation. Consider using method 2 or 3.")
            
            # Test basic model functionality
            try:
                test_input = torch.tensor([[1]]).to(device)
                with torch.no_grad():
                    test_output = model(input_ids=test_input, return_dict=True)
                print("Basic model forward pass successful")
                
                # Test if model can handle image inputs
                try:
                    test_image = torch.randn(1, 3, 224, 224).to(device)
                    with torch.no_grad():
                        test_output = model(input_ids=test_input, pixel_values=test_image, return_dict=True)
                    print("Basic model image forward pass successful")
                except Exception as e2:
                    try:
                        with torch.no_grad():
                            test_output = model(input_ids=test_input, images=test_image, return_dict=True)
                        print("Basic model image forward pass successful (using 'images' parameter)")
                    except Exception as e3:
                        print(f"Warning: Basic model image forward pass failed: {e3}")
                        print("This might indicate an issue with the model's image processing capabilities")
                        
            except Exception as e:
                print(f"Warning: Basic model forward pass failed: {e}")
            
            break
        except Exception as e:
            print(f"Method {i} failed: {e}")
            if i == len(loading_methods):
                print("All loading methods failed!")
                print("This might be due to:")
                print("1. Insufficient GPU memory")
                print("2. PyTorch version compatibility issues")
                print("3. Model file corruption")
                print("4. Network connectivity issues")
                raise e
    
    # Load tokenizer for text processing
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Tokenizer loaded successfully")
    
    # Load processor for image processing
    print("Loading processor...")
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        print("Processor loaded successfully with official template approach")
    except Exception as e:
        print(f"Warning: Could not load AutoProcessor: {e}")
        print("Falling back to manual image processing...")
        processor = None

    from transformers import Qwen2VLImageProcessor
    slow_image_processor = Qwen2VLImageProcessor.from_pretrained(model_name,use_fast=False)
    print("slow_image_processor ready:", type(slow_image_processor))

    # Optional checks and self-test; non-fatal
    if processor is not None:
        try:
            from transformers import Qwen2VLImageProcessor
            print("slow_image_processor cls:", type(slow_image_processor))
            print("is_fast attr:", getattr(slow_image_processor, "is_fast", None))
            if not isinstance(slow_image_processor, Qwen2VLImageProcessor):
                print("Warning: Not using Qwen2VLImageProcessor")
            if getattr(slow_image_processor, "is_fast", False):
                print("Warning: image processor appears to be fast; prefer use_fast=False")
        except Exception as e:
            print(f"Warning: Processor detail check failed: {e}")

        try:
            test_image_path = 'adversarial_images/test_image.jpeg'
            test_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": test_image_path},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
            text = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(test_messages)
            _ = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            print("Processor test successful with official template approach")
        except Exception as e:
            print(f"Warning: Processor self-test failed: {e}")
            print("Proceeding with processor anyway.")
    
    # Ensure model is on the correct device and can compute gradients
    if hasattr(model, 'device_map') and model.device_map is not None:
        print("Model loaded with device_map, ensuring it's on the correct device...")
        # For models with device_map, we need to ensure they're accessible
        if device != "cpu":
            print("Moving model to specified device...")
            try:
                model = model.to(device)
                print(f"Model moved to {device}")
            except Exception as e:
                print(f"Warning: Could not move model to {device}: {e}")
                print("Model will use device_map instead")
    
    # Check for any optimizations that might interfere with gradients
    if hasattr(model, 'config'):
        config = model.config
        if hasattr(config, 'use_cache'):
            print(f"Model use_cache setting: {config.use_cache}")
        if hasattr(config, 'gradient_checkpointing'):
            print(f"Model gradient_checkpointing setting: {config.gradient_checkpointing}")
    
    # Ensure model is properly set up for gradient computation
    print("Setting up model for gradient computation...")
    model.eval()  # Start in eval mode
    model.requires_grad_(True)  # Enable gradients for all parameters
    
    # Test if model can compute gradients
    try:
        assert processor is not None, "Processor is None"
        # If you already built test_inputs above, reuse it; else build once.
        if "test_inputs" not in locals():
            # Reuse test_messages you already created above
            assert "test_messages" in locals(), "test_messages missing"
            text = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(test_messages)
            test_inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            print("Built test_inputs on the fly for grad test")

        # Move to device once
        inputs = {k: v.to(device) for k, v in test_inputs.items()}

        # Prep training-style forward (so loss has grad)
        was_training = model.training
        use_cache_was = getattr(getattr(model, "config", None), "use_cache", None)
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model.train()

        with torch.enable_grad():
            # Prefer supervised loss if input_ids exist
            if "input_ids" in inputs:
                labels = inputs["input_ids"].clone()
                pad_id = getattr(getattr(model, "config", None), "pad_token_id", None)
                if pad_id is None and hasattr(processor, "tokenizer"):
                    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
                if pad_id is not None:
                    labels[labels == pad_id] = -100

                outputs = model(**inputs, labels=labels, return_dict=True)
                loss = outputs.loss
            else:
                # Fallback: make a scalar from logits
                outputs = model(**inputs, return_dict=True)
                if getattr(outputs, "logits", None) is None:
                    raise RuntimeError("No logits and no input_ids; cannot form a loss.")
                loss = outputs.logits.float().sum()

            print(f"Dummy loss: {loss.item():.6f}, requires_grad={loss.requires_grad}")
            assert loss.requires_grad, "Loss has no grad; broken gradient path."
            loss.backward()

        print("Model gradient computation test successful")
        
    except Exception as e:
        print(f"Warning: Model gradient computation test failed: {e}")
        print("This might cause issues during the attack")
    
    return model, tokenizer, processor, slow_image_processor

def process_image_for_model(image_path, processor, device):
    """Process image to tensor format compatible with Qwen2.5-VL model"""
    # Load image
    from qwen_vl_utils import process_vision_info
    image = Image.open(image_path).convert('RGB')
    
    if processor is not None:
        try:
            # Try official template approach first
            if 'process_vision_info' in globals() and process_vision_info is not None:
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": "What do you see in this image?"},
                            ],
                        }
                    ]
                    
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    processed = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    print("[Qwen2.5-VL] Used official template approach")
                    
                except Exception as e:
                    print(f"[Qwen2.5-VL] Official template approach failed: {e}")
                    print("[Qwen2.5-VL] Falling back to direct processor approach")
                    processed = processor(
                        images=image, 
                        text="What do you see in this image?",
                        return_tensors='pt'
                    )
            else:
                # Fallback to direct processor approach
                processed = processor(
                    images=image, 
                    text="What do you see in this image?",
                    return_tensors='pt'
                )
            
            if processed is None or "pixel_values" not in processed:
                raise ValueError("processor returned None or missing 'pixel_values'")
            
            image_tensor = processed["pixel_values"].to(device)
            print(f"[Qwen2.5-VL] processor output shape: {tuple(image_tensor.shape)}")
            
            # Ensure correct shape: [B, C, H, W] where C=3 for RGB
            if len(image_tensor.shape) == 3:
                # [C, H, W] -> [1, C, H, W]
                image_tensor = image_tensor.unsqueeze(0)
                print(f"[Qwen2.5-VL] Added batch dimension: {tuple(image_tensor.shape)}")
            elif len(image_tensor.shape) == 2:
                # [H, W] -> [1, 3, H, W] - convert to RGB
                image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                print(f"[Qwen2.5-VL] Added batch and RGB channels: {tuple(image_tensor.shape)}")
            
            # Ensure we have RGB channels (C=3)
            if image_tensor.shape[1] == 1:
                # Convert grayscale to RGB by repeating the channel
                image_tensor = image_tensor.repeat(1, 3, 1, 1)
                print(f"[Qwen2.5-VL] Converted grayscale to RGB: {tuple(image_tensor.shape)}")
            
            print(f"[Qwen2.5-VL] final shape: {tuple(image_tensor.shape)}")
            return image_tensor
            
        except Exception as e:
            print(f"Warning: Qwen2.5-VL processor failed: {e}")
            print("Falling back to manual processing...")

    # Fallback: manual processing with proper RGB channels
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)),  # Standard size for vision models
        T.ToTensor(),  # This converts to [C, H, W] format with C=3 for RGB
    ])
    
    # Apply transform: PIL Image -> [C, H, W] where C=3 for RGB
    image_tensor = transform(image)  # Shape: [3, 224, 224]
    
    # Add batch dimension: [C, H, W] -> [B, C, H, W]
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]
    
    print(f"[Fallback] manual processing, final shape: {tuple(image_tensor.shape)}")
    print(f"[Fallback] RGB channels: {image_tensor.shape[1]}")
    
    # Final validation: ensure correct shape and RGB channels
    assert len(image_tensor.shape) == 4, f"Image tensor must have 4D shape, got {image_tensor.shape}"
    assert image_tensor.shape[0] == 1, f"Batch dimension must be 1, got {image_tensor.shape[0]}"
    assert image_tensor.shape[1] == 3, f"Channel dimension must be 3 for RGB, got {image_tensor.shape[1]}"
    
    print(f"✅ Final image tensor: shape={tuple(image_tensor.shape)}, device={image_tensor.device}, dtype={image_tensor.dtype}")
    return image_tensor

# ========================================
#             Model Initialization
# ========================================

print('>>> Initializing Qwen2.5-VL Model')

args = parse_args()

# Set device
device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}')

# Check PyTorch version and handle compatibility
torch_version = torch.__version__
print(f"PyTorch version: {torch_version}")

# Handle PyTorch version compatibility
if torch_version.startswith("2.0") or torch_version.startswith("2.1"):
    print("Warning: PyTorch 2.0/2.1 detected. Some features may not work properly.")
    # Disable compilation features that might cause issues
    if hasattr(torch, 'compiler'):
        try:
            torch.compiler.disable()
            print("Disabled torch.compiler to avoid compatibility issues")
        except:
            print("Could not disable torch.compiler")
    # Set additional environment variables for older PyTorch versions
    os.environ["PYTORCH_DISABLE_COMPILER"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
elif torch_version.startswith("1."):
    print("PyTorch 1.x detected. Using legacy mode.")
    # Disable newer features
    os.environ["PYTORCH_DISABLE_COMPILER"] = "1"
elif torch_version.startswith("2.2") or torch_version.startswith("2.3"):
    print("PyTorch 2.2+ detected. Should be compatible.")
else:
    print(f"Unknown PyTorch version: {torch_version}")

# Additional compatibility fixes
try:
    # Disable JIT profiling that might cause issues
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    print("Disabled JIT profiling")
except:
    print("Could not disable JIT profiling")

# Try to disable torch.compile if available
try:
    if hasattr(torch, 'compile'):
        # Override torch.compile to do nothing
        original_compile = torch.compile
        def dummy_compile(*args, **kwargs):
            return args[0] if args else None
        torch.compile = dummy_compile
        print("Disabled torch.compile to avoid compatibility issues")
except:
    print("Could not disable torch.compile")

# Additional environment variable settings for transformers
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Try to patch transformers to avoid torch.compiler issues
try:
    import transformers
    if hasattr(transformers, 'utils'):
        # Disable any compilation features in transformers
        if hasattr(transformers.utils, 'is_torch_compile_available'):
            transformers.utils.is_torch_compile_available = lambda: False
        if hasattr(transformers.utils, 'is_torch_fx_available'):
            transformers.utils.is_torch_fx_available = lambda: False
        print("Patched transformers to disable compilation features")
except Exception as e:
    print(f"Could not patch transformers: {e}")

# Load model and tokenizer
try:
    model, tokenizer, processor, slow_image_processor = get_model_and_tokenizer(args.model_name, device)
    model.eval()
    print('[Model Initialization Finished]\n')
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check if the model name is correct and you have sufficient GPU memory.")
    exit(1)

# Create output directory
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    print(f"Created output directory: {args.save_dir}")

# Check if template image directory exists
template_dir = os.path.dirname(args.template_img)
if template_dir and not os.path.exists(template_dir):
    print(f"Warning: Template image directory '{template_dir}' does not exist.")
    print("Please ensure the template image path is correct.")

# Check if harmful corpus directory exists
corpus_dir = os.path.dirname(args.harmful_corpus)
if corpus_dir and not os.path.exists(corpus_dir):
    print(f"Warning: Harmful corpus directory '{corpus_dir}' does not exist.")
    print("Please ensure the harmful corpus path is correct.")

# Load harmful corpus
print('Loading harmful corpus...')
file = open(args.harmful_corpus, "r")
data = list(csv.reader(file, delimiter=","))
file.close()
targets = []
num = len(data)
for i in range(num):
    targets.append(data[i][0])

print(f'Loaded {len(targets)} harmful targets')

# Load and process template image
template_img = args.template_img
image_tensor = process_image_for_model(template_img, processor, device)
print(f"Image shape: {image_tensor.shape}")

# Test basic gradient computation
print("Testing basic gradient computation...")
test_tensor = image_tensor.clone().detach().requires_grad_(True)
print(f"Test tensor requires_grad: {test_tensor.requires_grad}")

# Simple gradient test
test_output = test_tensor.sum()
test_output.backward()
print(f"Gradient test successful: grad shape: {test_tensor.grad.shape}")

# Test model gradient computation BEFORE initializing attacker
print("Testing model gradient computation...")
test_image = image_tensor.clone().detach().requires_grad_(True)
print(f"Test image requires_grad: {test_image.requires_grad}")

# Test model mode switching
print(f"Initial model training mode: {model.training}")
model.train()
print(f"Model set to training mode: {model.training}")
model.eval()
print(f"Model set to eval mode: {model.training}")


# Gradient test using processor-prepared multimodal inputs
if processor is None:
    raise RuntimeError("Processor is None; cannot build multimodal inputs for gradient test")

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

if process_vision_info is None:
    raise RuntimeError("process_vision_info is unavailable; cannot run official multimodal pipeline for gradient test")


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "adversarial_images/test_image.jpeg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

with torch.enable_grad():
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    model_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move to device and enable gradients on pixel_values for image perturbation
    for key, value in model_inputs.items():
        model_inputs[key] = value.to(device)
    if "pixel_values" not in model_inputs or model_inputs["pixel_values"] is None:
        raise RuntimeError("processor did not return 'pixel_values' for gradient test")
    model_inputs["pixel_values"].requires_grad_(True)

    outputs = model(**model_inputs, return_dict=True)
    loss = outputs.logits.sum()
    print(f"Gradient test loss: {loss}")
    loss.backward()
    print(f"Model gradient test successful: pixel_values grad shape: {model_inputs['pixel_values'].grad.shape}")


# Initialize attacker AFTER gradient test
from qwen_utils import visual_attacker

print('device = ', device)
if processor is not None:
    my_attacker = visual_attacker.Attacker(
        args, model, tokenizer, targets, 
        device=device, image_processor=slow_image_processor
    )
else:
    print("Warning: No image processor available, initializing attacker without it")
    my_attacker = visual_attacker.Attacker(
        args, model, tokenizer, targets, 
        device=device
    )

# Prepare text prompt
from qwen_utils import prompt_wrapper
text_prompt_template = prompt_wrapper.prepare_text_prompt('What do you see in this image?')
print(f"Text prompt template: {text_prompt_template}")

# Run attack
try:
    if not args.constrained:
        print('[Unconstrained Attack]')
        adv_patch_vectors = my_attacker.attack_unconstrained(
            text_prompt_template,
            img=image_tensor, 
            batch_size=8,
            num_iter=args.n_iters, 
            alpha=args.alpha/255
        )
    else:
        print('[Constrained Attack]')
        adv_patch_vectors = my_attacker.attack_constrained(
            text_prompt_template,
            img=image_tensor, 
            batch_size=8,
            num_iter=args.n_iters, 
            alpha=args.alpha/255,
            epsilon=args.eps/255
        )
    
    # Convert patch vectors back to image format for saving
    print("Converting patch vectors back to image format...")
    
    # Get the parameters needed for conversion (from the last processor call)
    proc_out = my_attacker.image_processor(
        images=[Image.open(args.template_img).convert('RGB')],
        return_tensors='pt',
        data_format='channels_first'
    )
    grid = proc_out["image_grid_thw"]
    mean = torch.tensor(my_attacker.image_processor.image_mean, device=my_attacker.device).view(1, 1, 3, 1, 1)
    std = torch.tensor(my_attacker.image_processor.image_std, device=my_attacker.device).view(1, 1, 3, 1, 1)
    p = int(my_attacker.image_processor.patch_size)
    tps = int(getattr(my_attacker.image_processor, 'temporal_patch_size', 1))
    
    # Convert to image
    adv_img = visual_attacker.Attacker.patch_vectors_to_image(
        adv_patch_vectors, grid, mean, std, p, tps
    )
    
    # Save adversarial image
    save_image(adv_img, f'{args.save_dir}/qwen_adversarial_image.bmp')
    print('[Attack Completed]')
    print(f'Adversarial image saved to: {args.save_dir}/qwen_adversarial_image.bmp')
    
except Exception as e:
    print(f"Error during attack execution: {e}")
    print("Attack failed. Please check the error message above.")
    exit(1)
