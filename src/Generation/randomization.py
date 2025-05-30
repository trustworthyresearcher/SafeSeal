import random
import torch
import numpy as np

def randomize(original, alternatives, similarity):
    """
    Randomly select.

    Args:
        original: The original word (string).
        alternatives: List of alternative words (list of strings). [A, B, C]

    Returns:
        A randomized word, which could be the original or one of its alternatives.
    """
    choice = [original] + alternatives
    randomized_word = random.choice(choice)

    return randomized_word