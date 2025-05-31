# Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions

This repository contains the official implementation for our ICML paper submission titled "Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions". 

Note: Due to the large size of the results files, we are unable to include them in the submission. However, we will provide them upon request. We plan to release the evaluation models for HarmScore after the review period ends.

## Table of Contents

- [Description](#description)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
  - [Speak Easy](#speakeasy)
  - [Translators](#translators)
  - [Frameworks](#frameworks)
- [Metrics](#metrics)

## Description
This repository provides the implementation of our proposed **Speak Easy** method for jailbreaking LLMs. Our approach uses simple multi-step and multilingual querying to elicit sufficiently harmful responses. 

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

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

Ensure that you have access to necessary APIs (e.g., OpenAI API) and have configured the appropriate authentication tokens.

## Repository Structure

Here is a brief description of the contents of each folder and file in the repository:

- `backbones`: Code for loading and defining LLM backbones (currently supports Ollama, OpenAI models, and vLLM models).
- `data`: Datasets for benchmarking jailbreak efficacy and test sets for evaluating response selection models.
- `eval_models`: Implementations for measuring and evaluating harmfulness, including ASR and HarmScore.
- `frameworks`: Implementations of all attack frameworks tested in our paper (e.g., DR + Speak Easy, GCG + Speak Easy).
- `resp_select_models`: Code for HarmScore to load and compute actionability and informativeness.
- `results`: Outputs of frameworks and evaluation results.
- `translation`: Code for loading and using translation modules.
- `utils`: Constants and helper functions.

## Usage

### Speak Easy

You can use our Speak Easy framework either independently (on top of direct requests) or in combination with existing jailbreak methods such as GCG. To run the framework, execute:

```
python run_frameworks.py --data-dir [DATASET] \
                         --save-dir results/ \
                         --model [BACKBONE] \
                         --frameworks [FRAMEWORK]
```

- `[DATASET]`: Path to a JSON file containing queries, formatted as:

  ```
  [
      { "query": "query1" },
      { "query": "query2" },
      ...
  ]
  ```

- `[BACKBONE]`: Specifies the model to use, in the format `[MODEL_FAMILY]:[MODEL]`, e.g., `openai:gpt-4`.

- `[FRAMEWORK]`: The attack framework to use. For a list of available frameworks, refer to the `get_framework()` function in `frameworks/__init__.py`.

### Translators

Our framework supports multiple translation modules to facilitate multilingual attacks. To load a translator:

```
from translation import get_translator

translator = get_translator("[TRANSLATOR_NAME]")
```

Supported translators include:

- [Azure Translator API](https://learn.microsoft.com/en-us/azure/ai-services/translator/)
- Google Translate via the [`deep-translator`](https://pypi.org/project/deep-translator/) library
- [Google Cloud Translation API](https://cloud.google.com/translate)

### Frameworks

We implement `Speak Easy` on top of the following frameworks:

- **DR (Direct Request)**
- **[GCG (Greedy Coordinate Gradient)](https://github.com/llm-attacks/llm-attacks)**
  - To replicate our experiments with GCG, you can use the default suffixes provided in the `frameworks/baseline/gcg/config`.yaml. Alternatively, you may choose to regenerate them if needed.
- **[TAP (Tree of Attacks with Pruning)](https://github.com/RICommunity/TAP)**
  - To replicate our experiments with TAP, re-run the attack frameworks of `baseline_tap` or `speakeasy_tap` .

## Metrics

We provide implementations for evaluating the attack success and harmfulness of generated outputs. Run the following script specifying the evaluation metric you want to use.


```
from eval_models import get_metric

eval_model = get_metric("gpt4",device=0)
OR
eval_model = get_metric("harmscore", device)
```

---