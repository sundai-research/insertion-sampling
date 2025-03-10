import asyncio
from hashlib import sha256
import os
import numpy as np
import ray
import torch
from numba import njit

        
@njit
def stopping_criteria_fn(output_tokens, delimiter_tokens):
    count = 0
    for d in delimiter_tokens:
        count += np.sum(output_tokens == d)
    return count


@ray.remote
class AbortingVLLMWorker:
    """
    Base vLLM worker class encapsulating common logic.
    Inference requires an input_data dict and additional **kwargs to update SamplingParams.
    Subclasses must implement:
      - get_engine_args(...) to provide engine configuration.
    Generation workers allow dynamic generation parameters (temperature, n, max tokens),
    while logprob workers enforce fixed parameters for single token generation.
    """
    def __init__(self, model_path: str, worker_id: str,
                 tensor_parallel_size: int = 1, max_num_seqs: int = 16, delimiter_str: str = "\n\n"):
        from vllm import AsyncLLMEngine
        from transformers import AutoTokenizer
        self.counter = 0
        self.model_path = model_path
        self.worker_id = worker_id
        print(f"Initializing {self.__class__.__name__} with model path: {model_path}")
        self.engine_args = self.get_engine_args(model_path, tensor_parallel_size, max_num_seqs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = AsyncLLMEngine.from_engine_args(self.engine_args, start_engine_loop=True)
        self.delimiter_tokens = get_delimiter_tokens_list(self.tokenizer, delimiter_str)
        # self.registry = get_or_create_registry("generation_vllm_registry")
        # self.setup_registration()
    
    def get_engine_args(self, model_path: str, tensor_parallel_size: int, max_num_seqs: int):
        from vllm import AsyncEngineArgs
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        return AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.98,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            max_num_seqs=max_num_seqs,
            max_model_len=config.max_position_embeddings,
            disable_log_requests=True,
            disable_log_stats=True,
        )
    
    def parse_sample(self, sample: dict, output_ids, num_delimiters: int=0):
        sample['output_token_ids'] = list(output_ids)
        sample['output_len'] = len(sample['output_token_ids'])
        sample['sample_ids'] = sample['input_token_ids'] + sample['output_token_ids']
        sample['sample_text'] = self.tokenizer.decode(sample['sample_ids'])
        sample['sample_position_ids'] = list(range(len(sample['sample_ids'])))
        sample['num_delimiters'] = num_delimiters
        return sample
    
    def get_max_tokens(self, sample: dict, max_tokens=None) -> int:
        max_tokens = max_tokens if max_tokens is not None else self.engine_args.max_model_len
        max_tokens = max_tokens - len(sample['input_token_ids']) - 1
        if max_tokens <= 0:
            max_tokens = 1
            print(f"\033[1;38;2;255;165;0mMax tokens is less than 0 for sample: \033[0m {sample['input']}")
        return max_tokens
    
    def get_gen_kwargs(self, sample: dict, **kwargs) -> dict:
        from vllm import SamplingParams
        return SamplingParams(**{
            "n": kwargs.get("n", 1),
            "max_tokens": self.get_max_tokens(sample, kwargs.get("max_tokens", self.engine_args.max_model_len)),
            "temperature": kwargs.get("temperature", 0.7),
            "include_stop_str_in_output": kwargs.get("include_stop_str_in_output", False),
            "spaces_between_special_tokens": False,
            "skip_special_tokens": False,
        })
    
    async def inference(self, sample: dict, 
                        stopping_criteria_fn: callable = stopping_criteria_fn,
                        num_delimiters: int = 10,
                        **kwargs) -> list[dict]:
        """
        Base inference:
          - Input: input_data must include 'input_token_ids'.
          - **kwargs can include extra parameters.
          - Constructs a SamplingParams object and then performs inference.
        """
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        try:
            assert kwargs.get("n", 1) is not None
            input_ids = sample['input_token_ids']
            sampling_params = self.get_gen_kwargs(sample, **kwargs)
            request_id = sha256((str(sampling_params) + str(input_ids) + str(self.counter))
                                .encode('utf-8')).hexdigest()
            self.counter += 1
            print(f"[{self.__class__.__name__} {self.worker_id}] Sampling params: {sampling_params}")
            generator = self.llm.generate(
                prompt=TokensPrompt(prompt_token_ids=input_ids),
                sampling_params=sampling_params,
                request_id=request_id
            )
            # delimiter_tokens = np.array(self.tokenizer.encode(delimiter_str))
            output_ids = None
            abort_task = None
            async for out in generator:
                output_ids = out.outputs[0].token_ids
                out_num_delimiters = stopping_criteria_fn(np.array(output_ids), self.delimiter_tokens)
                if out_num_delimiters >= num_delimiters:
                    abort_task = asyncio.create_task(self.llm.abort(request_id))
                    return self.parse_sample(sample, output_ids, num_delimiters)
            if abort_task is not None:
                await abort_task
            return self.parse_sample(sample, output_ids)
        except Exception as e:
            print(f"\033[38;5;196m\033[1mError during inference for worker {self.worker_id}: {e}\033[0m")
            import traceback
            traceback.print_exc()
            raise e
        
def get_delimiter_tokens_list(tokenizer, delimiter_str: str) -> list[int]:
    tokenizer_vocab = tokenizer.vocab
    delimiter_id = tokenizer.encode(delimiter_str)
    assert len(delimiter_id) == 1
    delimiter_str = [k for k,v in tokenizer_vocab.items() if v == delimiter_id[0]][0]
    print(f"delimiter_str: {delimiter_str}")
    result = []
    for k,v in tokenizer_vocab.items():
        if k.endswith(delimiter_str):
            result.append(v)
    return np.array(result)

def get_runtime_env(mode: str):
    runtime_env = {"env_vars": dict(os.environ)}
    runtime_env["env_vars"].pop("CUDA_VISIBLE_DEVICES", None)
    if mode == "generation":
        runtime_env["pip"] = [
            f"-r {os.path.join(os.path.dirname(__file__), 'requirements_vllm.txt')}",   
            "math-verify[antlr4_13_2]"
        ]
        runtime_env["excludes"] = ["*.pyc", "__pycache__"]
    elif mode == "logprob":
        runtime_env["pip"] = [f"-r {os.path.join(os.path.dirname(__file__), 'requirements_fsdp.txt')}"]
    return runtime_env
        
if __name__ == "__main__":
    ray.init(address="auto", namespace="test")
    model_path = "/dev/shm/Qwen2.5-1.5B"
    worker = AbortingVLLMWorker.options(
        runtime_env=get_runtime_env("generation"),
        num_gpus=1,
        num_cpus=1,
    ).remote(model_path, 
             "aborting_vllm_worker", 
             1, 
             16,
             delimiter_str="\n\n")
    # worker = AbortingVLLMWorker(model_path, "aborting_vllm_worker", 1, 16)
    from transformers import AutoTokenizer
    import nest_asyncio
    nest_asyncio.apply()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    text = "Please reason step by step, and put your final answer within \\boxed{}. How would you compute the volume of a shape made of the union of two spheres of the same radius where the centers coincide with the surface of the other sphere?, please"
    token_ids = tokenizer.encode(text)
    sample = {
        'input_token_ids': token_ids,
    }
    result = ray.get(worker.inference.remote(sample, stopping_criteria_fn=stopping_criteria_fn, num_delimiters=2))
    print(result['sample_text'])
    from IPython import embed
    embed()
    delimiter_tokens = get_delimiter_tokens_list(tokenizer, "\n\n")
    stopping_criteria_fn(np.array(result['output_token_ids']), delimiter_tokens, 1)
    total = 0
    for d in delimiter_tokens:
        total += np.sum(np.array(result['output_token_ids']) == d)
    print(f"total: {total}")


    delimiter_tokens = get_delimiter_tokens_list(tokenizer, "\n\n")
    for d in delimiter_tokens:
        print(rf"{tokenizer.decode(d)}")
    print(f"delimiter_tokens: {delimiter_tokens}")
            

    