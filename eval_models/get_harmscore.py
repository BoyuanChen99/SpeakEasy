import os
import json
import argparse
from eval_models import get_metric

device = 0 
eval_model = get_metric("harmscore", device)

AVERAGE_SCORES_FILE = "average_harm_scores.json"

def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'.")
        return None


def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def process_file(json_file):
    """
    Process a single JSON file containing query-response pairs.
    
    Parameters:
        json_file (str): Path to the input JSON file.
    """
    print(f"Processing file: {json_file}")

    data = load_json(json_file)
    if data is None:
        return

    formatted_data = {query: [{"test_case": query, "generation": response}] for query, response in data.items()}

    reward_scores = eval_model.compute_scores(formatted_data, os.path.basename(json_file))
    average_score = eval_model.compute_average_score(reward_scores)

    print(f"{json_file}: Average HarmScore = {average_score}")

    existing_scores = load_json(AVERAGE_SCORES_FILE) or {}
    existing_scores[json_file] = average_score
    save_json(existing_scores, AVERAGE_SCORES_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate responses using HarmScore.")
    parser.add_argument("input_json", help="Path to the input JSON file containing query-response pairs.")

    args = parser.parse_args()
    
    if os.path.isfile(args.input_json):
        process_file(args.input_json)
    else:
        print(f"Error: '{args.input_json}' is not a valid file.")