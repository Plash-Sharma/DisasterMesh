"""
Save/load experiment results.
Ref: Mega-Prompt Evaluation
"""
import numpy as np

def save_results(results, path):
    np.save(path, results)

def load_results(path):
    return np.load(path, allow_pickle=True)
