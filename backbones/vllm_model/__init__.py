import os
import json
import torch 

import ray
from vllm import LLM, SamplingParams
from tqdm import tqdm
from munch import Munch

from transformers import AutoTokenizer
from huggingface_hub import login as hf_login

class VLLMModel:
    def __init__(self, model_name_or_path, num_gpus=None, hf_token=None):
        """
        Initialize the VLLM model with configurations and GPU settings.
        """
        if hf_token:
            hf_login(hf_token)
        
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()

        self.model_name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if num_gpus > 1:
            ray.init(ignore_reinit_error=True)
            resources = ray.cluster_resources()
            available_devices = ",".join([str(i) for i in range(int(resources.get("GPU", 0)))])
            os.environ['CUDA_VISIBLE_DEVICES'] = available_devices

        params = Munch.fromYAML(open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r"))
        self.temperature = params.temperature
        self.top_p = params.top_p
        self.max_tokens = params.max_tokens

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )

        self.llm = LLM(model=model_name_or_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.95, dtype=torch.float16, max_model_len=4096)

    def infer_batch(self, inputs, save_dir, batch_size=1):
        """
        Perform batch inference on the input data.

        Args:
            inputs (list): Input data in string format.
            save_dir (str): Path to save the output responses.
            batch_size (int): Number of samples to process in a batch.

        Returns:
            list: The complete list of responses.
        """
        responses = (
            json.load(open(save_dir, "r")) if os.path.exists(save_dir) else list()
        )
        start_index = len(responses)

        for i in tqdm(range(start_index, len(inputs), batch_size), desc="Processing"):
            batch_inputs = inputs[i:i + batch_size]
            messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": batch_inputs[0]}]

            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            try:
                outputs = self.llm.generate([prompt], self.sampling_params)
            except Exception as e:
                print(f"Error during generation: {e}")
                outputs = []

            # Parse outputs
            for output_idx, input_text in enumerate(batch_inputs):
                if outputs and output_idx < len(outputs) and outputs[output_idx].outputs:
                    responses.append(outputs[output_idx].outputs[0].text)
                else:
                    responses.append("")

            json.dump(responses, open(save_dir, "w"), indent=4)

        return responses