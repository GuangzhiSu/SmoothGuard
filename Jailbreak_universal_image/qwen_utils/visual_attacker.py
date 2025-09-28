import torch
from tqdm import tqdm
import random
from qwen_utils import prompt_wrapper, generator
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import seaborn as sns
from PIL import Image
from qwen_vl_utils import process_vision_info
import inspect

# Qwen2.5-VL uses different normalization values
def normalize(images):
    # Qwen2.5-VL uses ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(images.device)
    # Ensure gradient preservation
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    # Qwen2.5-VL uses ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(images.device)
    # Ensure gradient preservation
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


class Attacker:

    def __init__(self, args, model, tokenizer, targets, device='cuda:0', is_rtp=False, image_processor=None):

        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.is_rtp = is_rtp

        self.targets = targets
        self.num_targets = len(targets)

        self.loss_buffer = []

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)

        self.image_processor = image_processor

        # Debug model configuration
        print("=== Model Configuration Debug ===")
        print(inspect.signature(model.forward))
        if hasattr(model, 'config'):
            cfg = model.config
            print("Model config:", cfg)
            if hasattr(cfg, 'vision_config'):
                print("Vision config:", cfg.vision_config)
                print("in_channels:", cfg.vision_config.in_channels)
            print("vocab_size:", cfg.vocab_size)
            print("hidden_size:", cfg.hidden_size)
        
        # Debug image processor configuration
        if image_processor is not None:
            print("=== Image Processor Debug ===")
            print("Image processor:", image_processor)
            if hasattr(image_processor, 'image_processor'):
                ip = image_processor.image_processor
                print("Inner processor:", ip)
                print("do_rescale:", ip.do_rescale)
                print("do_normalize:", ip.do_normalize)
                print("image_mean:", ip.image_mean, "image_std:", ip.image_std)
                print("expected channels:", getattr(ip, "num_channels", getattr(ip, "channels", "unknown")))
        print("=" * 50)

    
    @staticmethod
    def _repeat_first_dim(x, B):
        # Repeat along batch dim if tensor has a leading dim of 1
        if isinstance(x, torch.Tensor) and x.dim() >= 1 and x.size(0) == 1:
            return x.repeat(B, *([1] * (x.dim() - 1)))
        return x

    @staticmethod
    def _ensure_batched(tensor_or_value, batch_size: int):
        """Ensure the first dim is batch_size for torch tensors.
        - If 2D (T,D) -> (B,T,D)
        - If 3D and first dim==1 -> (B,*,*)
        - If 4D and first dim==1 -> (B,*,*,*)
        Non-tensors returned as-is.
        """
        if not isinstance(tensor_or_value, torch.Tensor):
            return tensor_or_value
        t = tensor_or_value
        if t.dim() == 0:
            # scalar, expand to (B,)
            return t.view(1).repeat(batch_size)
        if t.dim() == 1:
            # (D) -> (B,D)
            return t.unsqueeze(0).repeat(batch_size, 1)
        if t.dim() == 2:
            # (T,D) -> (B,T,D)
            return t.unsqueeze(0).repeat(batch_size, 1, 1)
        if t.size(0) == 1:
            # (1,...) -> (B,...)
            reps = [batch_size] + [1] * (t.dim() - 1)
            return t.repeat(*reps)
        # If already batched and matches B, return as-is
        if t.size(0) == batch_size:
            return t
        # Mismatch: safest is to raise for visibility
        raise ValueError(f"Tensor first dim {t.size(0)} != batch_size {batch_size} and != 1: shape {tuple(t.shape)}")

    @staticmethod
    def _repeat_to_batch(value, batch_size: int):
        """Repeat various container types to match batch size.
        - torch.Tensor: use _ensure_batched
        - list/tuple: if len==1, replicate; else if len==batch_size, return; else error
        - dict: apply recursively to values
        - others: return as-is
        """
        if isinstance(value, torch.Tensor):
            return Attacker._ensure_batched(value, batch_size)
        if isinstance(value, dict):
            return {k: Attacker._repeat_to_batch(v, batch_size) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            if len(value) == batch_size:
                return value
            if len(value) == 1:
                return type(value)(value * batch_size)
            raise ValueError(f"List/Tuple length {len(value)} incompatible with batch_size {batch_size}")
        return value

    def _debug_token_bounds(self, input_ids, labels=None):
        try:
            vocab_size = getattr(self.tokenizer, 'vocab_size', None)
            if vocab_size is None:
                vocab_size = getattr(getattr(self.model, 'config', object()), 'vocab_size', None)
            if isinstance(input_ids, torch.Tensor):
                min_id = int(input_ids.min().item())
                max_id = int(input_ids.max().item())
                print(f"[debug] input_ids range: [{min_id}, {max_id}], vocab_size={vocab_size}")
                if vocab_size is not None:
                    assert max_id < vocab_size and min_id >= 0, \
                        f"input_ids out of range: [{min_id},{max_id}] vs vocab_size={vocab_size}"
            if labels is not None and isinstance(labels, torch.Tensor):
                masked = labels[labels != -100]
                if masked.numel() > 0:
                    min_l = int(masked.min().item())
                    max_l = int(masked.max().item())
                    print(f"[debug] labels (!= -100) range: [{min_l}, {max_l}], vocab_size={vocab_size}")
                    if vocab_size is not None:
                        assert max_l < vocab_size and min_l >= 0, \
                            f"labels out of range: [{min_l},{max_l}] vs vocab_size={vocab_size}"
        except Exception as e:
            print(f"[debug] token bounds check skipped due to: {e}")

    @staticmethod
    def clamp_patch_like_image(pv, grid, mean, std, p, tps):
        """
        pv: (Np, tps*3*p*p)  ->  reshape to (T',H',W',tps,3,p,p),
        clamp values to normalized range [ (0-mean)/std, (1-mean)/std ] by channel, then flatten back.
        Note: T',H',W' are only used for shape; for static images T'=1.
        """
        try:
            # Ensure grid has correct shape
            if grid.dim() == 2 and grid.shape[1] == 3:
                # (1, 3) or (B, 3)
                T_, H_, W_ = grid[0].tolist()
            else:
                raise ValueError(f"Expected grid shape (1,3) or (B,3), got {tuple(grid.shape)}")
            
            # Verify pv shape is compatible with grid
            expected_patches = T_ * H_ * W_
            if pv.shape[0] != expected_patches:
                raise ValueError(f"Expected {expected_patches} patches based on grid {[T_, H_, W_]}, but got {pv.shape[0]}")
            
            x = pv.view(T_, H_, W_, tps, 3, p, p)          # (T',H',W',tps,3,p,p)
            
            # Ensure mean and std have correct shape for broadcasting
            if mean.dim() == 5:  # (1,1,3,1,1)
                lo = (0.0 - mean) / std
                hi = (1.0 - mean) / std
            else:
                # If other shapes are passed, reshape them
                mean_reshaped = mean.view(1, 1, 3, 1, 1)
                std_reshaped = std.view(1, 1, 3, 1, 1)
                lo = (0.0 - mean_reshaped) / std_reshaped
                hi = (1.0 - mean_reshaped) / std_reshaped
                
            # broadcast to (T',H',W',tps,3,p,p) channel dimension
            x = torch.maximum(x, lo) 
            x = torch.minimum(x, hi)
            return x.view(-1, tps*3*p*p)
            
        except Exception as e:
            print(f"Error in clamp_patch_like_image: {e}")
            print(f"pv shape: {tuple(pv.shape)}")
            print(f"grid shape: {tuple(grid.shape)}")
            print(f"mean shape: {tuple(mean.shape)}")
            print(f"std shape: {tuple(std.shape)}")
            print(f"p: {p}, tps: {tps}")
            raise e
    
    @staticmethod
    def patch_vectors_to_image(patch_vectors, grid, mean, std, p, tps):
        """
        Convert patch vectors back to image format for saving
        patch_vectors: (Np, tps*3*p*p) or (B, Np, tps*3*p*p)
        grid: (1, 3) or (B, 3)
        mean, std: normalization parameters
        p: patch_size
        tps: temporal_patch_size
        returns: (B, 3, H, W) or (3, H, W) image tensor
        """
        try:
            # Handle batch dimension
            if patch_vectors.dim() == 2:
                # (Np, D) -> (1, Np, D)
                patch_vectors = patch_vectors.unsqueeze(0)
                grid = grid.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
                
            B, Np, D = patch_vectors.shape
            
            # Get spatial dimensions from grid
            if grid.dim() == 2 and grid.shape[1] == 3:
                T_, H_, W_ = grid[0].tolist()
            else:
                raise ValueError(f"Expected grid shape (1,3) or (B,3), got {tuple(grid.shape)}")
            
            # Verify dimension compatibility
            expected_patches = T_ * H_ * W_
            if Np != expected_patches:
                raise ValueError(f"Expected {expected_patches} patches based on grid {[T_, H_, W_]}, but got {Np}")
            
            expected_dim = tps * 3 * p * p
            if D != expected_dim:
                raise ValueError(f"Expected patch dimension {expected_dim}, but got {D}")
            
            # Reshape to (B, T', H', W', tps, 3, p, p)
            x = patch_vectors.view(B, T_, H_, W_, tps, 3, p, p)
            
            # Rearrange dimensions to (B, 3, H, W)
            x = x.permute(0, 5, 1, 3, 2, 4, 6)  # (B, 3, T', p, H', p, W')
            x = x.reshape(B, 3, T_ * p, H_ * p)  # (B, 3, H, W)
            
            # Denormalize
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)
            x = x * std + mean
            
            # Clamp to [0, 1] range
            x = torch.clamp(x, 0, 1)
            
            if squeeze_output:
                x = x.squeeze(0)  # (3, H, W)
                
            return x
            
        except Exception as e:
            print(f"Error in patch_vectors_to_image: {e}")
            print(f"patch_vectors shape: {tuple(patch_vectors.shape)}")
            print(f"grid shape: {tuple(grid.shape)}")
            print(f"mean shape: {tuple(mean.shape)}")
            print(f"std shape: {tuple(std.shape)}")
            print(f"p: {p}, tps: {tps}")
            raise e

    def build_supervised_text_batch(self, user_text: str, targets: list):
        """
        Construct each sample as: [PROMPT TOKENS] + [TARGET TOKENS + eos]
        input_ids: concatenated sequence of both
        labels:    PROMPT part set to -100, TARGET part equals target token ids
        """
        B = len(targets)
        prompt_ids = self.tokenizer(user_text, add_special_tokens=True).input_ids  # e.g., [bos, ..., eos]
        # If you don't want prompt to include eos automatically, use add_special_tokens=False to control template

        input_ids_list, labels_list = [], []
        for t in targets:
            tgt_ids = self.tokenizer(t, add_special_tokens=False).input_ids
            # Ensure ending with eos (many instruction tuning paradigms do this)
            if self.tokenizer.eos_token_id is not None:
                tgt_ids = tgt_ids + [self.tokenizer.eos_token_id]

            ids = prompt_ids + tgt_ids
            lbs = [-100] * len(prompt_ids) + tgt_ids

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(lbs, dtype=torch.long))

        # pad to same length
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
        }

    def attack_unconstrained(self, text_prompt, img, batch_size=8, num_iter=2000, alpha=1/255):
        print('>>> batch_size:', batch_size)
        print(f'>>> Input image shape: {img.shape}, requires_grad: {img.requires_grad}')

        # 1) Normalize input to (1,3,H,W)
        if img.ndim != 4:
            if img.ndim == 2:
                img = img.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            elif img.ndim == 3:
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                img = img.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
        assert img.shape[0] == 1 and img.shape[1] == 3, f"expect (1,3,H,W), got {img.shape}"
        print(f'>>> Validated input image shape: {img.shape}')

        # 2) Get patch vectors pixel_values and image_grid_thw through processor
        assert self.image_processor is not None, "Qwen2.5-VL needs image_processor"
        img_tensor = img.squeeze(0)
        if img_tensor.max() <= 1.0:
            img_uint8 = (img_tensor.clamp(0,1) * 255).byte()
        else:
            img_uint8 = img_tensor.clamp(0,255).byte()
        img_pil = Image.fromarray(img_uint8.permute(1,2,0).cpu().numpy())

        proc_out = self.image_processor(
            images=[img_pil],
            return_tensors='pt',
            data_format='channels_first'
        )
        base_pv = proc_out.get("pixel_values", None).to(self.device)  # Expected shape: (N_patches, tps*3*p*p)
        grid = proc_out["image_grid_thw"].to(self.device)
        print(f">>> pixel_values (patch vectors) shape: {tuple(base_pv.shape)}; grid: {tuple(grid.shape)}")

        # Related normalization and patch hyperparameters
        mean = torch.tensor(self.image_processor.image_mean, device=self.device).view(1, 1, 3, 1, 1)
        std  = torch.tensor(self.image_processor.image_std,  device=self.device).view(1, 1, 3, 1, 1)
        p    = int(self.image_processor.patch_size)
        tps  = int(getattr(self.image_processor, 'temporal_patch_size', 1))

        # 3) Adversarial variables directly defined in patch vector domain
        adv_pv = base_pv.clone().detach().requires_grad_(True)  # (Np, D)

        prompt = prompt_wrapper.Prompt(self.model, self.tokenizer, text_prompts=text_prompt, device=self.device)

        for t in tqdm(range(num_iter + 1)):
            batch_targets = random.sample(self.targets, batch_size)

            # Forward & backward pass (model expects pixel_values=patch vectors, with image_grid_thw)
            loss = self.attack_loss_qwen(
                prompt,
                adv_pv,
                {"image_grid_thw": grid},
                batch_targets,
            )
            loss.backward()

            # Unconstrained PGD: gradient descent then project to valid pixel range
            with torch.no_grad():
                adv_pv.add_(-alpha * adv_pv.grad.sign())
                adv_pv.copy_(Attacker.clamp_patch_like_image(adv_pv, grid, mean, std, p, tps))
                adv_pv.grad.zero_()
                self.model.zero_grad()

            self.loss_buffer.append(loss.item())
            if t % 20 == 0:
                self.plot_loss()

        # Return final adversarial patch vectors
        return adv_pv.detach().cpu()


    
    def attack_constrained(self, text_prompt, img, batch_size=8, num_iter=2000,
                       alpha=1/255, epsilon=128/255):
        print('>>> batch_size:', batch_size)
        print(f'>>> Input image shape: {img.shape}, requires_grad: {img.requires_grad}')

        # 1) Normalize input to (1,3,H,W)
        if img.ndim != 4:
            if img.ndim == 2:
                img = img.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            elif img.ndim == 3:
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                img = img.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
        assert img.shape[0] == 1 and img.shape[1] == 3, f"expect (1,3,H,W), got {img.shape}"
        print(f'>>> Validated input image shape: {img.shape}')

        # 2) Do pixel preprocessing once to get base_pixels=(1,3,H,W)
        assert self.image_processor is not None, "Qwen2.5-VL needs image_processor"
        img_tensor = img.squeeze(0)  # (3,H,W)
        if img_tensor.max() <= 1.0:
            img_uint8 = (img_tensor.clamp(0,1) * 255).byte()
        else:
            img_uint8 = img_tensor.clamp(0,255).byte()
        img_pil = Image.fromarray(img_uint8.permute(1,2,0).cpu().numpy())

        proc_out = self.image_processor(
            images=[img_pil],
            return_tensors='pt',
            data_format='channels_first'
        )
        base_pv = proc_out.get("pixel_values", None).to(self.device)  # (Np, D)
        grid = proc_out["image_grid_thw"].to(self.device)
        print(f">>> pixel_values (patch vectors) shape: {tuple(base_pv.shape)}; grid: {tuple(grid.shape)}")

        # Related normalization and patch hyperparameters
        mean = torch.tensor(self.image_processor.image_mean, device=self.device).view(1, 1, 3, 1, 1)
        std  = torch.tensor(self.image_processor.image_std,  device=self.device).view(1, 1, 3, 1, 1)
        p    = int(self.image_processor.patch_size)
        tps  = int(getattr(self.image_processor, 'temporal_patch_size', 1))

        # 3) L_inf perturbation variables (patch vector domain)
        adv_delta = torch.empty_like(base_pv).uniform_(-epsilon, epsilon).to(self.device)
        adv_delta.requires_grad_(True)
        adv_delta.retain_grad()

        prompt = prompt_wrapper.Prompt(self.model, self.tokenizer, text_prompts=text_prompt, device=self.device)

        for t in tqdm(range(num_iter + 1)):
            batch_targets = random.sample(self.targets, batch_size)

            # Current adversarial patch vectors (first project to valid pixel domain)
            adv_cur = base_pv + adv_delta
            adv_cur = Attacker.clamp_patch_like_image(adv_cur, grid, mean, std, p, tps)

            # Forward & backward pass (patch vectors + grid)
            loss = self.attack_loss_qwen(
                prompt,
                adv_cur,
                {"image_grid_thw": grid},
                batch_targets,
            )
            loss.backward()

            # PGD update and project back to L_inf ball and pixel domain
            with torch.no_grad():
                adv_delta.add_(-alpha * adv_delta.grad.sign())
                adv_delta.clamp_(-epsilon, epsilon)
                # Project assembled adv_cur back to pixel domain range
                adv_cur = base_pv + adv_delta
                adv_cur = Attacker.clamp_patch_like_image(adv_cur, grid, mean, std, p, tps)
                # Align adv_delta with projected adv_cur (for next iteration)
                adv_delta.copy_(adv_cur - base_pv)

                adv_delta.grad.zero_()
                self.model.zero_grad()

            self.loss_buffer.append(loss.item())
            if t % 20 == 0:
                self.plot_loss()

        # Final adversarial patch vectors
        final_adv = Attacker.clamp_patch_like_image(base_pv + adv_delta, grid, mean, std, p, tps)
        return final_adv.detach().cpu()


    def attack_loss_qwen(self, prompts, adv_vis, vision_meta, targets):
        # Build supervised text inputs and labels
        B = len(targets)
        user_text = prompts.text if hasattr(prompts, "text") else "Describe the image."
        text_inputs = self.build_supervised_text_batch(user_text, targets)

        # Repeat/add batch to vision inputs/meta
        adv_vis_b = self._ensure_batched(adv_vis, B)
        vision_meta_b = {k: self._repeat_to_batch(v, B) for k, v in vision_meta.items()}
        if isinstance(adv_vis_b, torch.Tensor) and adv_vis_b.dim() == 3:
            adv_vis_b = adv_vis_b.reshape(-1, adv_vis_b.size(-1))

        if "image_grid_thw" in vision_meta_b:
            g = vision_meta_b["image_grid_thw"]
            # Could be (1,3) or transformed to (B,1,3) by _ensure_batched
            if isinstance(g, torch.Tensor):
                if g.dim() == 2 and g.shape == (1, 3):
                    g = g.repeat(B, 1)       # -> (B,3)
                elif g.dim() == 3 and g.shape[1] == 1 and g.shape[2] == 3:
                    g = g.squeeze(1)         # (B,1,3) -> (B,3)
                vision_meta_b["image_grid_thw"] = g

        # Debug batch structure
        print("=== Qwen Attack Loss Batch Debug ===")
        print("Batch size:", B)
        print("Adv_vis shape:", tuple(adv_vis_b.shape) if isinstance(adv_vis_b, torch.Tensor) else type(adv_vis_b))
        print("Vision meta:")
        for k, v in vision_meta_b.items():
            try:
                print(f"  {k}: {tuple(v.shape)}")
            except Exception:
                print(f"  {k}: {type(v)}")
        print("Text inputs:")
        for k, v in text_inputs.items():
            try:
                print(f"  {k}: {tuple(v.shape)}")
            except Exception:
                print(f"  {k}: {type(v)}")
        print("Labels shape:", tuple(text_inputs["labels"].shape))
        print("=" * 40)

        # Sanity checks to avoid index out of bounds inside model
        def _first_dim(x):
            return x.size(0) if isinstance(x, torch.Tensor) and x.dim() >= 1 else None
                # Sanity checks to avoid index out of bounds inside model

        def _validate_adv_and_meta(adv, grid_b, B):
            # grid_b: (B,3)
            assert isinstance(grid_b, torch.Tensor) and grid_b.dim() == 2 and grid_b.shape[1] == 3, \
                f"image_grid_thw must be (B,3), got {tuple(grid_b.shape)}"
            T_, H_, W_ = map(int, grid_b[0].tolist())
            Np = T_ * H_ * W_

            if isinstance(adv, torch.Tensor):
                if adv.dim() == 2:
                    # Two valid cases after flattening: single sample (Np,D) or batch flattened (B*Np,D)
                    assert adv.size(1) > 0, f"adv last dim invalid: {adv.size(1)}"
                    assert adv.size(0) in (Np, B * Np), \
                        f"adv first dim {adv.size(0)} not in (Np={Np}, B*Np={B*Np})"
                elif adv.dim() == 3:
                    # Not flattened: should be (B, Np, D)
                    assert adv.size(0) == B and adv.size(1) == Np and adv.size(2) > 0, \
                        f"adv shape should be (B,Np,D)=({B},{Np},D), got {tuple(adv.shape)}"
                else:
                    raise ValueError(f"Unexpected adv dim: {adv.dim()}")
            else:
                raise ValueError("adv must be a torch.Tensor")

        _validate_adv_and_meta(adv_vis_b, vision_meta_b["image_grid_thw"], B)
        
        for k, v in vision_meta_b.items():
            fd = _first_dim(v)
            if fd is not None:
                assert fd in (None, B), f"vision meta '{k}' batch mismatch: {fd} vs {B}"

        # Forward + loss
        self.model.zero_grad()
        with torch.enable_grad():
            was_training = self.model.training
            self.model.train()
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

            try:
                outputs = self.model(
                    **text_inputs,
                    pixel_values=adv_vis_b,
                    labels=text_inputs["labels"],
                    return_dict=True,
                    **vision_meta_b,
                )
            except Exception as e:
                # Debug dump
                print("[attack_loss_qwen] Forward error:", e)
                print("- text_inputs keys:", list(text_inputs.keys()))
                for k,v in text_inputs.items():
                    print(f"  {k}: {tuple(v.shape)}")
                if isinstance(adv_vis_b, torch.Tensor):
                    print("- adv_vis_b shape:", tuple(adv_vis_b.shape))
                for k,v in vision_meta_b.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  meta {k}: {tuple(v.shape)}")
                    else:
                        print(f"  meta {k}: type={type(v)}")
                # Retry with 'images' kw for older signatures
                outputs = self.model(
                    **text_inputs,
                    images=adv_vis_b,
                    return_dict=True,
                    **vision_meta_b,
                )

            loss = outputs.loss
            self.model.train(was_training)

        assert loss.requires_grad, "loss has no gradient (image/vision variables not in computation graph)"
        return loss




    def plot_loss(self):

        sns.set_theme()
        num_iters = len(self.loss_buffer)

        x_ticks = list(range(0, num_iters))

        # Plot and label the training and validation loss values
        plt.plot(x_ticks, self.loss_buffer, label='Target Loss')

        # Add in a title and axes labels
        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')

        # Display the plot
        plt.legend(loc='best')
        plt.savefig('%s/loss_curve.png' % (self.args.save_dir))
        plt.clf()

        torch.save(self.loss_buffer, '%s/loss' % (self.args.save_dir))

    def attack_loss(self, prompts, images, targets):
        # images: (B,3,H,W), and is adv_pixels.repeat(B,1,1,1), requires_grad=True
        B = len(targets)
        assert images.shape[0] == B and images.shape[1] == 3, f"expect (B,3,H,W), got {tuple(images.shape)}"

        # 1) Process text only with processor (don't pass images)
        user_text = prompts.text if hasattr(prompts, "text") else "Describe the image."
        tok_text = self.tokenizer([user_text]*B, padding=True, return_tensors='pt')
        text_inputs = {k: v.to(self.device) for k, v in tok_text.items()}  # input_ids / attention_mask

        # 2) targets → labels
        tok = self.tokenizer(targets, padding=True, return_tensors='pt').to(self.device)
        labels = tok.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Debug token id bounds to detect CUDA index issues early
        self._debug_token_bounds(text_inputs.get('input_ids', None), labels)

        # Debug batch structure
        print("=== Batch Debug ===")
        print("Batch size:", B)
        print("Images shape:", tuple(images.shape))
        print("Text inputs:")
        for k, v in text_inputs.items():
            try:
                print(f"  {k}: {tuple(v.shape)}")
            except Exception:
                print(f"  {k}: {type(v)}")
        print("Labels shape:", tuple(labels.shape))
        print("=" * 30)

        # 3) Forward pass (key: directly feed pixels to model; no longer use processor(images=...))
        self.model.zero_grad()
        with torch.enable_grad():
            was_training = self.model.training
            self.model.train()
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

            # Compatible with different signatures: prefer pixel_values=, fallback to images=
            try:
                outputs = self.model(
                    **text_inputs,
                    pixel_values=images,
                    labels=labels,
                    return_dict=True,
                )
            except TypeError:
                outputs = self.model(
                    **text_inputs,
                    images=images,
                    labels=labels,
                    return_dict=True,
                )

            loss = outputs.loss
            self.model.train(was_training)

        assert loss.requires_grad, "loss has no gradient (image not in computation graph). Please confirm processor(images=...) is not called."
        return loss