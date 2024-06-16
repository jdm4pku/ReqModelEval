import os
import json
"""
读取原始文件，然后划分成句子.
"""
def preprocess(input,output):
   for filename in os.listdir(input):
      if filename.endswith('.txt'):
         file_path = os.path.join(input,filename)
         with open(file_path,'r',encoding='utf-8') as file:
            lines = file.readlines()
         processed_lines = []
         current_sentence = ""
         for line in lines:
            line = line.strip()
            if line:
               current_sentence += line + " "
               if line.endswith(('.','!','?')): ## 句子结束的标志
                  cur_sens = current_sentence.strip().split('.')[:-1]
                  for item in cur_sens:
                     processed_lines.append(item.strip()+'.')
                  current_sentence = ""
         # if current_sentence: # 处理最后一行
         #    processed_lines.append(current_sentence.strip())
         output_path = os.path.join(output,filename)
         with open(output_path,'w',encoding='utf-8') as file:
            file.write('\n'.join(processed_lines))


def main():
   input_dir = '/home/jindongming/project/modeling/ReqModelEval/unlabel_req/unprocess'
   output_dir = "/home/jindongming/project/modeling/ReqModelEval/unlabel_req/process"
   preprocess(input_dir,output_dir)


if __name__=="__main__":
   main()







            