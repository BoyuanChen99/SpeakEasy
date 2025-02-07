# Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions

This repository includes the code used in the [**Speak Easy**](https://arxiv.org/abs/2502.04322v1) paper. We show that simple interactions such as multi-step, multilingual querying can elicit sufficiently harmful jailbreaks from LLMs. We also design a metric, HarmScore, to measure the actionability and informativeness of jailbreak responses. 

## Dependencies
The primary dependencies are:

```
torch
ollama
transformers
openai
huggingface-cli
box
tqdm
```

For a comprehensive list of required libraries and their versions, please refer to the [`requirements.txt`](requirements.txt) file.
Ensure that you have access to necessary APIs (e.g., OpenAI API) and have configured the authentication tokens.


## Repository Structure
For safety reasons, we are not publicly releasing the code for implementing the Speak Easy framework at this time. However, we do plan to release it at a later date. If you would like access now, please contact the corresponding author of the paper.

The current repository contains: 
- `backbones`: Code for loading and defining LLM backbones (currently supports Ollama, OpenAI models, and vLLM models).
- `data`: Datasets with harmful instructions for benchmarking jailbreak methods.
- `eval_models`: Implementations for measuring and evaluating harmfulness, including ASR and HarmScore.
- `utils`: Constants and helper functions.

To be released: 
- `frameworks`: Implementations of all attack frameworks tested in our paper (e.g., DR + Speak Easy, GCG + Speak Easy).


## Evaluation with HarmScore

To use HarmScore to evaluate the actionability and informativeness (and thus, harmfulness) of jailbreak responses, you can provide your input data as a JSON file with the following **query-response** format:

```json
{
    "query_1": "response_1",
    "query_2": "response_2"
}

Each query is mapped to a corresponding generated response. To run the evaluation, use the following command in your terminal:

```
python3 evaluate.py path/to/input.json

```
The script will compute harm scores for each response and save the average score in average_harm_scores.json.


---