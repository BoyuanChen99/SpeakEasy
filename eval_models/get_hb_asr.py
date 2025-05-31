import os
import json

from eval_models import get_metric

reward_model = get_metric("harmbench", device=0)

def process_folder(input_file):
    data = json.load(open(input_file, 'r'))

    behaviors, generations = [], []
    for b, g in data.items():
        behaviors.append(b)
        generations.append(g)

    scored_data = reward_model.compute_scores(behaviors, generations)
    
    success_count = scored_data.count('Yes')
    asr = 0 if success_count == 0 else success_count / len(scored_data)

    print(success_count)
    print(len(scored_data))

    print(f"{input_file}: {asr}")

process_folder("")