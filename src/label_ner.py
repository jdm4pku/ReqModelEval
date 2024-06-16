"""
利用Qwen2 标注命名实体数据集
"""
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os
import json

def get_completion(prompts, model, tokenizer=None, max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=2048):
    stop_token_ids = [151329, 151336, 151338]
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len,trust_remote_code=True,gpu_memory_utilization=0.9,tensor_parallel_size=4)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def label_by_llm(input_dir,output_dir,prompt_path,llm_name):
    with open(prompt_path,'r',encoding='utf-8') as file:
        prompt_template = file.read()
    for filename in os.listdir(input_dir):
        label_data = []
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir,filename)
            with open(file_path,'r',encoding='utf-8') as file:
                lines = file.readlines()
            for line in tqdm(lines,desc=f"Processing {filename}"):
                prompt = prompt_template.format(input_req=line)
                result = get_completion(prompt,llm_name)
                item = {
                    "input":result,
                    "output":result
                }
                label_data.append(item)
            json_data = json.dumps(label_data,ensure_ascii=False,indent=2)
            output_path = os.path.join(output_dir,f"{filename[-4]}.json")
            with open(output_path,'w',encoding='utf-8') as output_file:
                output_file.write(json_data)
                

def main():
    input_dir = "/home/jindongming/project/modeling/ReqModelEval/unlabel_req/process"
    ouput_dir = "/home/jindongming/project/modeling/ReqModelEval/dataset/PD/entity"
    prompt_path = "/home/jindongming/project/modeling/ReqModelEval/prompt/create-ner-prompt.txt"
    llm_name = "Qwen/Qwen2-7B-Instruct"
    label_by_llm(input_dir,ouput_dir,prompt_path,llm_name)


if __name__=="__main__":
    main()



































