import re
import random
import sys
sys.path.append('/proj/dmfexp/statllm/users/mikhail/sunday-ai/deepscaler/deepscaler/rewards/math_utils')
from utils import grade_answer_mathd, grade_answer_sympy, extract_answer # generated then reference

def math_reward(response, reference):
    
    model_answer = extract_answer(response)

    if model_answer is None:
        return 0.0

    if "\\boxed" in reference:
        reference = extract_answer(reference)
            
    if not reference:
        print('DELETE THIS SAMPLE')
        return 0.0

    # Check against all possible correct answers
    # for ground_truth in processed_ground_truths:
    is_correct = grade_answer_mathd(model_answer, reference) or grade_answer_sympy(model_answer, reference)
    if is_correct:
        return 1.0

    return 0.0
    

def insert_phrase(response, delimiter, special_phrases, position = None, max_position = 1000):
    """
    Modifies the response by finding all occurrences of the delimiter,
    choosing one occurrence at random, truncating the response at that point,
    and appending a random phrase from special_phrases.
    
    Parameters:
        response (str): The original string.
        delimiter (str): The delimiter to search for in response.
        special_phrases (list of str): A list of phrases to randomly append.
    
    Returns:
        str: The modified string.
    """
    # Find all indices where the delimiter occurs
    indices = []
    start = 0
    while True:
        index = response.find(delimiter, start)
        if index == -1:
            break
        indices.append(index)
        # Move past this occurrence so we can find further ones.
        start = index + len(delimiter)
    
    # If we found any delimiters, choose one at random and truncate the response.
    if indices:
        if position is None:
            chosen_index = random.choice(indices[:max_position])
        else:
            chosen_index = indices[position]
        # Option 1: If you want to discard the delimiter itself, use:
        truncated_response = response[:(chosen_index + len(delimiter))]
        # Option 2: If you want to keep the delimiter, uncomment the next line and comment the previous:
        # truncated_response = response[:chosen_index + len(delimiter)]
    else:
        # If no delimiter is found, just keep the full response.
        truncated_response = response + delimiter

    # Append a random phrase from special_phrases.
    random_phrase = random.choice(special_phrases)
    
    return truncated_response + random_phrase, len(indices)