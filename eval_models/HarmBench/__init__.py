import os

from munch import Munch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HarmBench:
    """
    Evaluate jailbreak responses using the trained classifier from 
    HarmBench (https://huggingface.co/cais/HarmBench-Llama-2-13b-cls). 
    """
    def __init__(self, model_name="cais/HarmBench-Llama-2-13b-cls"):
        self.config = Munch.fromYAML(
            open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r")
        )

        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, truncation_side="left", padding_side="left"
        )

    def compute_scores(self, behaviors, generations, batch_size=10):
        behaviors = [self.config.behavior_prompt + b for b in behaviors]
        final_out = []

        total_inputs = len(behaviors)
        indices = list(range(0, total_inputs, batch_size))

        for start in indices:
            end = start + batch_size
            inputs = [
                self.config.prompt.format(behavior=b, generation=g)
                for b, g in zip(behaviors, generations)
            ]
            inputs = inputs[start:end]  

            if not inputs:  
                continue

            encoded = self.tokenizer(inputs, max_length=2048, truncation=True, return_tensors="pt", padding="longest")
            
            encoded = encoded.to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **encoded.to(self.model.device),
                    do_sample=False,
                    max_new_tokens=1,
                ).cpu()
                output_ids = output_ids[:, len(encoded.input_ids[0]) :]

            completion = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            final_out.extend(completion)

            torch.cuda.empty_cache()

        return final_out
