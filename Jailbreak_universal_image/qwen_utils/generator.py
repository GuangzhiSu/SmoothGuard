import torch
from transformers import StoppingCriteria, StoppingCriteriaList

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


class Generator:

    def __init__(self, model, tokenizer, max_new_tokens=1024, temperature=0.2, device='cuda:0'):

        self.model = model
        self.device = device
        self.tokenizer = tokenizer

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Qwen2.5-VL uses different stop tokens
        self.stop_str = "<|endoftext|>"
        self.keywords = [self.stop_str]
        
        # Ensure model is on the correct device
        if hasattr(self.model, 'device'):
            if str(self.model.device) != str(self.device):
                print(f"Warning: Model device {self.model.device} != generator device {self.device}")
        else:
            print(f"Warning: Model has no device attribute, using generator device: {self.device}")

    def generate(self, prompt, image):

        input_ids = prompt.input_ids[0]

        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)

        # Check if we need to preserve gradients (during attack)
        if image.requires_grad:
            # During attack, we need to preserve gradients
            with torch.enable_grad():
                output_ids = self.model.generate(
                    input_ids,
                    images=image,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
        else:
            # Normal generation mode
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(self.stop_str):
            outputs = outputs[:-len(self.stop_str)]
        outputs = outputs.strip()

        return outputs
