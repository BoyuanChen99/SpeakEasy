import os
import json
import argparse

from eval_models import get_metric

eval_model = get_metric("gpt4",device=0)

def process_outputs(input_file, output_folder, prefix=None, log_file="gpt_asr.txt"):
    print(f"Processing file: {input_file}")
    data = json.load(open(input_file, 'r'))
    data = list(data.items())
    asr = eval_model.compute_scores(data, output_folder, prefix=prefix)
    if asr == None: 
        pass
    else:
        result = f"{input_file}: {asr}\n"
        print(result)
        with open(log_file, 'a') as f:  
            f.write(result)

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--benchmark", type=str, help="Benchmark to calculate ASR for.")
    parser.add_argument("--method", type=str, help="Method to calculate ASR for.")

    args = parser.parse_args()
    method_path = f"./results/{args.benchmark}/{args.method}"
    model_names = ["openai:gpt-4o"] # "qwen2:72b-instruct"
        
    for model in os.listdir(method_path):
        if model in model_names:
            print(f"Processing: {method_path}/{model}")
            file_path = os.path.join(method_path, model, "combined_responses.json")
            if os.path.exists(file_path): 
                process_outputs(file_path, output_folder=os.path.join(method_path, model))

if __name__ == "__main__":
    main()