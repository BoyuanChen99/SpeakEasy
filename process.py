import pandas as pd
import json
import os

def main():
    # Load the data
    dataset = "strongreject"
    data_dir = "./data"
    input_file = os.path.join(data_dir, f"{dataset}.csv")
    output_file = os.path.join(data_dir, f"{dataset}.json")
    df = pd.read_csv(input_file)
    df = df[["prompt"]]
    df.rename(columns={"prompt": "query"}, inplace=True)
    data = df.to_dict(orient="records")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()