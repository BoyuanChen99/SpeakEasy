import os
import json

from eval_models import get_metric

device = 0
eval_model = get_metric("harmscore", device)

average_scores_file = "average_harm_scores.json" 

average_scores = {}

def process_folder(input_folder):
    response_file_path = input_folder
    print(f"Processing file: {response_file_path}")
    data = json.load(open(response_file_path, 'r'))

    formatted_data = {}
    for b, g in data.items():
        query = []
        query.append({"test_case": b, "generation": g})
        formatted_data[b] = query

    reward_scores = eval_model.get_reward(formatted_data, os.path.basename(response_file_path))
    
    average_score = eval_model.get_average_score(reward_scores)

    average_scores[response_file_path] = average_score
    print(f"{response_file_path}: Average HarmScore = {average_score}")

    with open(average_scores_file, 'r+', encoding='utf-8') as outfile:
        try:
            data = json.load(outfile)  
        except json.JSONDecodeError:
            data = {}

        data.update(average_scores)  

        outfile.seek(0)
        json.dump(data, outfile, ensure_ascii=False, indent=4)
        outfile.truncate()  

folder_name = ""
for file in os.listdir(folder_name):
    file_path = os.path.join(folder_name, file)
    process_folder(file_path)