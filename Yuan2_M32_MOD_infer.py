from vllm import LLM, SamplingParams
import time
from transformers import AutoTokenizer, LlamaTokenizer

def run_Mixtral():
    tokenizer = LlamaTokenizer.from_pretrained("/mnt/beegfs2/maojunxiong/MOE/Yuan2-M32-MOD", add_eos_token=False, add_bos_token=False, eos_token='<eod>', trust_remote_code=True)
    tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)
    sampling_params = SamplingParams(max_tokens=100, temperature=1, top_p=0.01, top_k=1)

    llm = LLM(model="/mnt/beegfs2/maojunxiong/MOE/Yuan2-M32-MOD", tensor_parallel_size=2, disable_custom_all_reduce=True, max_num_seqs=1, enforce_eager=True, trust_remote_code=True, gpu_memory_utilization=0.9)#, kv_cache_dtype='fp8')
    
    prompts = ['写一篇春游作文']#,'珠穆朗玛峰有多高','长江有多长','规划一个武汉旅游攻略']
    outputs = []
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    total_tokens = 0

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        token = len(tokenizer.encode(generated_text))
        total_tokens += token
        print(f"Generated text: {generated_text!r}")

    print("inference_time:", (end_time - start_time))
    print("total_tokens:", total_tokens)

if __name__ == "__main__":
    run_Mixtral()

