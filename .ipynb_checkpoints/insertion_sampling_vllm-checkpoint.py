import os
import time
import sys
import numpy as np
from optparse import OptionParser

sys.path.append('/proj/dmfexp/statllm/users/mikhail/sunday-ai/sample-march-2nd')
from gen_utils import insert_phrase, math_reward

import time
import copy
import re
import json

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
    
def main():

    options = parse_args()
    print(options)
    
    idx_start = options.idx_start
    idx_end = options.idx_end
    limo = options.limo
    model = options.model
    
    iters = 5
    max_position = 10

    if model in ['qwen32']:
        n_sample = 16
        max_try = 8
    else:
        n_sample = 64
        max_try = 4
        
    if model in ['phi', 'qwen32']:
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

    llm = LLM(model=model_name, enable_prefix_caching = True, gpu_memory_utilization = 0.95,
             tensor_parallel_size=model_configs[model]['n_gpu'], disable_custom_all_reduce=model_configs[model]['n_gpu']>1))
    
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

        if model in ['phi', 'granite', 'qwen32']:
            prompt_d = tokenizer.apply_chat_template([{"role": "user", 'content': SYS_PROMPT + d['problem']}], tokenize=False, add_generation_prompt=True)
        else:
            prompt_d = SYS_PROMPT + d['problem']

        t_s = time.time()
        for it in range(iters):
            
            found_answer = False
            count_try = 0
            while not found_answer:
                
                completions_bon = llm.generate(prompt_d + cur_answer, sampling_params, use_tqdm=False)

                responses = [out.text for out in completions_bon[0].outputs]

                scores = []
                for r in responses:
                    if r is None:
                        scores.append(0.)
                    else:
                        scores.append(math_reward(cur_answer + r, d['answer']))
                
                if max(scores) == 1:
                    found_answer = True
                else:
                    count_try += 1
                    if count_try == max_try:
                        break
        
            if not found_answer:
                print(f'Faied at iter {it}')
                sample_success = False
                break
                
            idx_sol = scores.index(1)
            if it < iters - 1:
                cut_response, delimiter_count = insert_phrase(responses[idx_sol], delimiter, special_phrases, max_position = max_position)
                cur_answer = cur_answer + cut_response
                cur_answer = re.sub(r'\\boxed\{([^}]+)\}', r'\1', cur_answer)
                # print(f'Delimiter count {delimiter_count}')
            else:
                cur_answer = cur_answer + responses[idx_sol]
            print(f'Iteration {it} mean reward {np.mean(scores)}')
            # print(cur_answer)

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