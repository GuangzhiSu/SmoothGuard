import torch
from transformers import AutoTokenizer

def prepare_text_prompt(user_prompt):
    """
    Prepare text prompt for Qwen2.5-VL model
    Qwen2.5-VL uses a specific format with <image> token
    """
    qs = "<image>\n" + user_prompt
    return qs

# support batch implementation
class Prompt:
    # tokenization
    # turn to embeddings

    # padding? wait until targets have been appended
    # prepare labels? need to wait for targets

    def __init__(self, model, tokenizer, text_prompts=None, device='cuda:0'):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.text_prompts = text_prompts
        self.context_length = []
        self.input_ids = []
        self.do_tokenization(self.text_prompts)

    def do_tokenization(self, text_prompts):

        if text_prompts is None:
            self.input_ids = []
            self.context_length = []
            return

        # For Qwen2.5-VL, we need to handle the <image> token specially
        # The tokenizer will automatically handle the <image> token
        input_ids = self.tokenizer(text_prompts, return_tensors='pt').input_ids.to(self.device)

        self.input_ids = [input_ids]
        self.context_length = [input_ids.shape[1]]
