from vllm import LLM, SamplingParams
import time
from transformers import AutoTokenizer, LlamaTokenizer
import json

def run_Yuan():
    
    sampling_params = SamplingParams(max_tokens=512, temperature=1, top_p=0, top_k=1, stop="<eod>")

    llm = LLM(model="/mnt/beegfs2/maojunxiong/Yuan_Models/Yuan_hf_20240914-1-4/", tensor_parallel_size=1, disable_custom_all_reduce=True, max_num_seqs=1, enforce_eager=True, trust_remote_code=True, gpu_memory_utilization=0.99, cpu_offload_gb=20, max_model_len=4000)
    
    results = []
    with open('/mnt/beegfs2/maojunxiong/HumanEval-instructions.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            prompts = data.get('prompt')
            outputs = llm.generate(prompts, sampling_params)
            results.append(outputs[0])

    generated_data = []
    with open('/mnt/beegfs2/maojunxiong/vLLM_ALL_VERSION/vllm_v0.4.0/humaneval/samples.jsonl', 'r', encoding='utf-8') as file:
        for line, result in zip(file, results):
            data = json.loads(line)
            task_id = data.get('task_id')
            completion = data.get('completion')
            generated_text = result.outputs[0].text
            data['completion'] = generated_text
            generated_data.append({'task_id': task_id, 'completion': generated_text})


    with open('/mnt/beegfs2/maojunxiong/Yuan_v0.5.4_tp1_0914ckpt.jsonl', 'w', encoding='utf-8') as file:
        for data in generated_data:
            json.dump(data, file, ensure_ascii=False)
            file.write('\n')

if __name__ == "__main__":
    run_Yuan()

