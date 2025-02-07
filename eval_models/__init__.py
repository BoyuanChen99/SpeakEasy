import os
import sys


def get_metric(metric_name, device):
    if os.path.dirname(__file__) not in sys.path:
        sys.path.append(os.path.dirname(__file__))

    if "gpt4" in metric_name:
        from gpt4 import GPT4Judge
        return GPT4Judge()
    
    elif "harmscore" in metric_name:
        from HarmScore import HarmScore
        return HarmScore(device)

    elif "harmbench" in metric_name:
        from HarmBench import HarmBench
        return HarmBench()
    
    else:
        raise NotImplementedError