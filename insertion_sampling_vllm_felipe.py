import os
import time
import sys
import numpy as np
from optparse import OptionParser
sys.path.append('/proj/dmfexp/statllm/users/mikhail/sunday-ai/sample-march-2nd')
from gen_utils import insert_phrase, math_reward
import random
import time
import copy
import re
import json

import torch
import gc

from vllm import LLM, SamplingParams

sys.path.append('/proj/dmfexp/statllm/users/mikhail/sunday-ai/deepscaler/deepscaler/rewards/math_utils')
from utils import extract_answer

from transformers import AutoTokenizer

def parse_args():
    parser = OptionParser()
    
    parser.add_option("--idx_start", type="int", dest="idx_start")
    parser.add_option("--idx_end", type="int", dest="idx_end")
    parser.add_option("--limo", type="int", dest="limo")
    parser.add_option("--model", type="str", dest="model")
    
    (options, args) = parser.parse_args()

    return options

EVAL_SAMPLING_PARAMS = SamplingParams(max_tokens=1, logprobs=20)
BREAK_STRING = "\n\n"

def get_equally_spaced_elements(lst, k):
    n = len(lst)
    # Return empty list if k is not positive
    if k <= 0:
        return []
    # If k is greater than or equal to the number of elements, return the original list
    if k >= n:
        return lst
    # Special case: if k is 1, return the first element (or choose as desired)
    if k == 1:
        return [lst[0]]
    # Compute the spacing step
    step = (n - 1) / (k - 1)
    # Generate indices and select corresponding elements
    return [lst[int(round(step * i))] for i in range(k)]

def compute_correctness_scores(llm, eval_prompts, eval_sampling_params):
    """
    Generates evaluations using the provided LLM, processes the log probabilities,
    and returns the computed correct scores.

    Parameters:
        llm: An object with a `generate` method for generating evaluations.
        eval_prompts: A list or sequence of prompts to be used for evaluation.
        eval_sampling_params: Parameters controlling the evaluation sampling.

    Returns:
        A numpy array of correct scores computed as:
            (exp(sum of log probabilities for 'yes')) / (sum of exp(log probabilities for 'yes' and 'no'))
    """
    # Generate evaluations without progress bar
    evals = llm.generate(eval_prompts, eval_sampling_params, use_tqdm=False)
    
    # Extract the log probability objects from the evaluations
    logprobs_list = [list(eval.outputs[0].logprobs[0].values()) for eval in evals]
    
    # Compute summed exponentiated log probabilities for tokens 'yes' and 'no'
    logprobs = [
        [
            np.exp([l.logprob for l in logprob if s in l.decoded_token.lower()]).sum()
            for s in ['yes', 'no']
        ]
        for logprob in logprobs_list
    ]
    
    # Convert to numpy array and compute correct scores as ratio for 'yes'
    correct_scores = np.array(logprobs)
    correct_scores = correct_scores[:, 0] / correct_scores.sum(axis=1)
    
    return correct_scores
    
EVAL_RESPONSE_PROMPT = """You are an evaluator. Your task is to assess the answer provided for a given question. 
Determine whether the answer is clear, thorough, and correct. 
If the answer meets all these criteria, respond exclusively with the token "Yes"; otherwise, respond exclusively with "No". 
Do not include any additional text.

Question: {}

Answer: {}

Evaluation:"""

EVAL_PARTIAL_RESPONSE_PROMPT = """You are an evaluator. Your task is to assess the partial answer provided for a given question. 
Determine whether the partial answer is clear, thorough, and correct. 
If the answer meets all these criteria, respond exclusively with the token "Yes"; otherwise, respond exclusively with "No". 
Do not include any additional text. If the partial answer is not good enough, the respondent will be prompted to rethink the partial answer.

Question: {}

Partial answer: {}

Evaluation:"""

def main():

    options = parse_args()
    print(options)
    
    idx_start = options.idx_start
    idx_end = options.idx_end
    limo = options.limo
    model = options.model
    
    iters = 5
    max_position = 10

    if model in ['qwen32', 'qwen05']:
        n_sample = 16
        max_try = 8
    else:
        n_sample = 64
        max_try = 4

    MAX_PARTIAL_RESPONSES = n_sample
        
    if model in ['phi', 'qwen32', 'qwen05']:
        max_tokens = 2000
    else:
        max_tokens = 4000
    
    delimiter = '\n\n'
    special_phrases = ['Let\'s try another method to solve the problem:', 'Now, let\'s think again:', 'Wait', 'Let\'s doublecheck the work so far.', 'Alternatively', 
                   'Let\'s look at it from a different perspective:']

    SYS_PROMPT = """Let's think step by step and output the final answer within \\boxed{}. """
    
    # model_name = "meta-llama/Llama-3.3-70B-Instruct"
    # model_name = "meta-llama/Llama-3.1-8B"
    # model_name = "Qwen/Qwen2-1.5B"
    # model_name = 'Qwen/Qwen2.5-Math-7B'
    # model_name = 'Qwen/Qwen2.5-1.5B'
    # model_name = 'microsoft/phi-4'

    models = {
              'qwen05': {'name':'Qwen/Qwen2.5-0.5B-Instruct','n_gpu':1},
              'phi': {'name':'microsoft/phi-4', 'n_gpu': 1},
              'qwen': {'name':'Qwen/Qwen2.5-1.5B', 'n_gpu':1},
              'granite': {'name':'ibm-granite/granite-3.1-8b-instruct', 'n_gpu':1},
              'qwen32': {'name': 'Qwen/Qwen2.5-32B-Instruct', 'n_gpu': 4}
             }

    if model not in models:
        sys.exit('Unknown model')
    else:
        model_name = models[model]['name']
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    if limo:
        data_path = '/proj/dmfexp/statllm/users/mikhail/sunday-ai/LIMO/train/data/limo.json'
    else:
        data_path = '/proj/dmfexp/statllm/users/mikhail/sunday-ai/deepscaler/deepscaler/data/train/deepscaler.json'

    data_name = data_path.split('/')[-1][:-5]
    
    base_dir = f'./results-{data_name}-{model}/'
    
    try:
        os.makedirs(base_dir)
    except:
        pass

    save_path = base_dir + f'{data_name}-{model}-insertion-data_{idx_start}_{idx_end}.json'
    
    with open(data_path) as f:
        data = json.load(f)

    print('Running ' + data_path.split('/')[-1], f'data size {len(data)} with batsh size {idx_end - idx_start}')

    if limo:
        bad_idx = []
    else:
        bad_idx = [6303, 6328, 6336, 6408, 6415, 38209, 38211]
    
    data = data[idx_start:idx_end]
    idx_range = np.arange(idx_start, idx_end)
    
    new_data = []

    sampling_params = SamplingParams(max_tokens = max_tokens,
                                 n = n_sample,
                                 temperature = 0.7)

    llm = LLM(model=model_name,
              enable_prefix_caching = True,
              gpu_memory_utilization = 0.95,
              tensor_parallel_size=models[model]['n_gpu'],
              disable_custom_all_reduce=models[model]['n_gpu']>1)
    
    for i, d in enumerate(data):

        print('\n' + 20*'==' + f' {idx_range[i]} ' + 20*'==' + '\n')
        
        if idx_range[i] in bad_idx:
            print('SKIPPED - bad idx')
            continue

        if limo:
            d['problem'] = d['instruction']
            d['answer'] = extract_answer(d['output'])

        if d['answer'] is None:
            print('SKIPPED - none answer')
            continue
            
        sample_success = True
        
        cur_answer = ''

        if model in ['phi', 'granite', 'qwen32', 'qwen05']:
            prompt_d = tokenizer.apply_chat_template([{"role": "user", 'content': SYS_PROMPT + d['problem']}], tokenize=False, add_generation_prompt=True)
        else:
            prompt_d = SYS_PROMPT + d['problem']

        t_s = time.time()
        for it in range(iters):

            completions_bon = llm.generate(prompt_d + cur_answer, sampling_params, use_tqdm=False)
            responses = [out.text for out in completions_bon[0].outputs]

            ### Felipe's code block ###
            responses = [cur_answer + r for r in responses]
            scores = []
            for r in responses:
                if r is None:
                    scores.append(0.)
                else:
                    scores.append(math_reward(r, d['answer']))

            first_response = responses[0]
            responses = [r for r in responses if BREAK_STRING in r]
            
            if len(responses)>0:
            
                ### selecting the response we will continue with
                scores2 = []
                for r in responses:
                    if r is None:
                        scores2.append(0.)
                    else:
                        scores2.append(math_reward(r, d['answer']))
                        
                if np.sum(scores2)>0:
                    idx_promissing_response = np.argmax(scores2)
                else:
                    eval_prompts = []
                    for resp in responses:
                        eval_prompts.append(tokenizer.apply_chat_template([{"role": "user", 'content': EVAL_RESPONSE_PROMPT.format(d['problem'],resp)}],
                                                                          tokenize=False, add_generation_prompt=True))
                
                    idx_promissing_response = np.argmax(compute_correctness_scores(llm, eval_prompts, EVAL_SAMPLING_PARAMS))
                
                ### selecting the partial response we will append a special phrase
                partial_responses = ['']
                for r in responses[idx_promissing_response].split(BREAK_STRING):
                    partial_responses.append(partial_responses[-1]+BREAK_STRING+r)
                partial_responses = partial_responses[1:]
                for ii in range(len(partial_responses)):
                    partial_responses[ii] = partial_responses[ii][len(BREAK_STRING):]
                partial_responses = get_equally_spaced_elements(partial_responses,MAX_PARTIAL_RESPONSES)
                
                partial_eval_prompts = []
                for resp in partial_responses:
                    partial_eval_prompts.append(tokenizer.apply_chat_template([{"role": "user", 'content': EVAL_PARTIAL_RESPONSE_PROMPT.format(d['problem'],resp)}],
                                                                      tokenize=False, add_generation_prompt=True))

                idx_not_promissing_response = np.argmin(compute_correctness_scores(llm, partial_eval_prompts, EVAL_SAMPLING_PARAMS))
    
                if it < iters - 1:
                    cur_answer = partial_responses[idx_not_promissing_response] + "\n\n" + random.choice(special_phrases)
                    cur_answer = re.sub(r'\\boxed\{([^}]+)\}', r'\1', cur_answer)
                else:
                    cur_answer = responses[idx_promissing_response]
            else:
                cur_answer = first_response
                
            print(f'Iteration {it} mean reward {np.mean(scores)}')
            ##################

        if sample_success:
            new_d = copy.deepcopy(d)
            new_d['insertion_sample'] = cur_answer
            new_d['problem-input'] = prompt_d
            new_data.append(new_d)
            with open(save_path, 'w') as f:
                json.dump(new_data, f)

        print('Sample took', time.time() - t_s)
        sys.stdout.flush()

if __name__ == "__main__":
    main()