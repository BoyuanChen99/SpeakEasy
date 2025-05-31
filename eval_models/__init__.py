import os
import sys


def get_metric(metric_name, device):
    if os.path.dirname(__file__) not in sys.path:
        sys.path.append(os.path.dirname(__file__))

    if "harmbench" in metric_name:
        from HarmBench import HarmBench
        return HarmBench()
    
    elif "harmscore" in metric_name:
        from harmscore import HarmScore
        return HarmScore(device)

    elif "gpt4" in metric_name:
        from gpt4 import GPT4Judge
        return GPT4Judge()
    
    else:
        raise NotImplementedError