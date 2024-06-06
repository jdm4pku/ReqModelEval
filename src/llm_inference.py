from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from argparse import ArgumentParser
import os
import json
import tqdm

def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset",type=str,required=True)
    parser.add_argument("--task",type=str,required=True,choices=['entity','relation']) 
    parser.add_argument("--prompt",type=str,required=True,choices=['qa','label'])
    parser.add_argument('--examples',type=str,default=0)
    parser.add_argument("--model", type=str, default='deepseek-7b')
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--moda", type=str, default='greedy')
    parser.add_argument("--max_tokens", type=int, default=500)
    return parser.parse_args()

def load_model(model_name:str):
    if model_name.startswith("gemma-7b"):
        print("Loading gemma-7b")
        model_dir = "gemma-7b"
    elif model_name.startswith("llama3-8b"):
        print("Loading llama3-8b")
        model_dir = "meta-llama/Meta-Llama-3-8B"
    elif model_name.startswith("qwen1.5-7b"):
        print("Loading qwen1.5-7b")
        model_dir = "Qwen/Qwen1.5-7B"
    elif model_name.startswith("chatglm3-6b"):
        print("Loading chatglm3-6b")
        model_dir = "THUDM/chatglm3-6b"
    model = LLM(model=model_dir,trust_remote_code=True,gpu_memory_utilization=0.9,tensor_parallel_size=4)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model,tokenizer


def retrieve_context_length(model_name:str):
    if model_name.startswith("gemma-7b"):
        return 8192
    elif model_name.startswith("llama3-8b"):
        return 8192
    elif model_name.startswith("qwen1.5-7b"):
        return 32768
    elif model_name.startswith("chatglm3-6b"):
        return 8192


def generate_prompt(args,task,sample):
    context_window = retrieve_context_length(args.model)
    if task=="entity":
        task_descrip = "You are expert specialising in entity extraction. Please extract"

        
    
    

def inference(args,data_path,task,model,tokenizer,output_dir,sampling_params):
    if not os.path.exist(output_dir):
        os.mkdirs(output_dir)
    output_file = os.path.join(output_dir,f'{task}.jsonl')
    with open(data_path,'r',encoding='utf-8') as file:
        data = json.load(file)
        with open(output_file,'a') as f_out:
            for sample in tqdm(data):
                prompt_ids = generate_prompt(args,task,sample,tokenizer)
                # try:
                #     results = model.generate(prompt_token_ids=[prompt_ids],sampling_params=sampling_params,use_tqdm=False)
                #     pass

                
def main():
    args = parser_args()
    model,tokenizer = load_model(args.model)
    print("Loading model and tokenizer.")
    if args.moda == "greedy":
        sampling_param = SamplingParams(temperature=0.0,max_tokens=args.max_tokens,n=1)
    elif args.moda == 'sampling':
        sampling_param = SamplingParams(temperature=0.4,top_p=0.95,max_tokens=args.max_tokens,n=20)
    else:
        raise ValueError("Invalid moda")
    
    inference(args,args.dataset,args.task,model,tokenizer,args.output_dir,sampling_param)

if __name__=="__main__":
    main()