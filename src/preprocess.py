import os
import json

os_dir = "/home/jindongming/project/modeling/ReqModelEval/data/PD"
data_ner = "/home/jindongming/project/modeling/ReqModelEval/data/PD/entity"

def generate_ner_dataset(json_file_list):
   dataset = []
   for file in json_file_list:
      file_path = os.path.join(os_dir,file)
      with open(file_path,'r',encoding='utf-8') as file:
         data = json.load(file)
         for sample in data:
            entities_list = {
               "Machine Domain":[],
               "Physical Device":[],
               "Environment Entity":[],
               "Design Domain":[],
               "Requirements":[],
               "Shared Phenomena":[]
            }
            input_text = sample['data']['text']
            annos = sample['annotations'][0]['result']
            for entity in annos:
               if 'value' not in entity:
                  continue
               if 'labels' not in entity['value']:
                  continue
               label = entity['value']['labels'][0]
               text = entity['value']['text']
               entities_list[label].append(text)
            ground_truth = str(entities_list)
            sen_entity = {
               "input":input_text,
               "entity":ground_truth
            }
            dataset.append(sen_entity)
   return dataset

def save_dataset(data,path):
   data_json = json.dumps(data, ensure_ascii=False, indent=2)
   with open(path, 'w', encoding='utf-8') as output_file:
     output_file.write(data_json)

def main():
   train_file = ['2-CCTNS.json','3-THEMAS.json','4-ESE.json','5-SFS.json']
   test_file = ['1-DHSS.json','6-TCS.json','7-VLA-CMCS.json']
   train_ner = generate_ner_dataset(train_file)
   test_ner = generate_ner_dataset(test_file)
   train_ner_path = os.path.join(data_ner,"train.json")
   test_ner_path = os.path.join(data_ner,"test.json")
   save_dataset(train_ner,train_ner_path)
   save_dataset(test_ner,test_ner_path)

if __name__=="__main__":
   main()








            