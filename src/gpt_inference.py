from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from argparse import ArgumentParser

def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset",type=str,required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, default='deepseek-7b')
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--moda", type=str, default='greedy')
    parser.add_argument("--max_tokens", type=int, default=500)
    return parser.parse_args()