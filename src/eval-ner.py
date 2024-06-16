from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from argparse import ArgumentParser
import os
import json
import tqdm

def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--data",type=str,required=True)
    parser.add_argument("--task",type=str,required=True)

def main():
    args = parser_args()


if __name__=="__main__":
    main()