from typing import Optional, Sequence, Dict
from src.methods.base import BaseMethod
from src.datasets.base import BaseDataset
from src.utils.registry import registry
from src import ImageTxtSample, _OutputType
from typing import Tuple, Callable
import re
import warnings
import json
import yaml
import pandas as pd
import os
from typing import List, Dict, Any, Sequence, Optional, Tuple, Callable
##from unsloth import FastLanguageModel
#from unsloth.chat_templates import get_chat_template
#from transformers import TextStreamer


@registry.register_dataset()
class Mimic(BaseDataset):
    dataset_ids: Sequence[str] = ["mimic", "mimic-binary", "mimic-binary-factuality",
                                   "mimic-open-ended-factuality"]
    dataset_config: Optional[Dict[str,str]]={}

    mimic_template = "{question}?"
    mimic_binary_template = "{question}? Answer [Yes] or [No]."
    mimic_binary_factuality_template = "{gender} {race} {question}? Answer [Yes] or [No]."
    mimic_open_ended_factuality_template = "{gender} {race} {question}?"

    def __init__(self, dataset_id:str, method_hook:Optional[BaseMethod]=None,**kwargs)->None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        base_dir = os.getcwd()  
        self.image_dir = "/home/pathin/safety_llm/Trust-Medical-LVLM/data/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
        self.anno_dir = "/home/pathin/safety_llm/Trust-Medical-LVLM/data/data/physionet.org/files/mimic-cxr-jpg/2.0.0/modified_mimic_factuality.json"
        self.annotations = self.load_json(self.anno_dir)
        # self.filter_json()
        self.build_dataset()    


    def load_json(self, path):
        with open(path, "r") as f:
            json_data = json.load(f)
        
        return json_data

        # TODO: adding the filtering

    def filter_json(self):
        new_json = []
        for entry in self.json_data:
            if entry["binAnswer"] in ["Yes", "No"]:
                new_json.append(entry)
        self.json_data = new_json
        print(f"filtered for yes/no: {len(self.json_data)} data loaded" )

    def build_dataset(self):
        dataset = []
        for entry in self.annotations:
            dataSample = self.template_format(entry)
            if os.path.exists(dataSample["image_path"]): 
                dataset.append(dataSample)
            # else:
            #     print(f"Image not found: {dataSample['image_path']}")
        
        self.dataset = dataset
        print(f"{len(dataset)} data loaded")

    # def prompt_template(self, info:Dict, prompt_tempte: Callable)->str:
    #     return prompt_tempte(info["gender"], info["age"], info["race"], info["text"], info["pre_text"])
    
    # def get_chat_template(self, gender:str, age:int, race:str, text:str, pre_text:str)->str:
    #     build_prompt = f"{text} {pre_text} {gender} {age} {race}"
    #     return build_prompt

    def template_format(self, anno:Dict)-> ImageTxtSample:
        image_path_ = os.path.join(self.image_dir, anno["image"])
        text_ = anno["text"]
        race_ = anno["race"]
        gender_ = anno["gender"]

        if anno["binAnswer"] != "N/A":
            target_ = anno["binAnswer"]
        else:
            target_ = anno["answer"]
        extra_ = {
            "gender": anno["gender"],
            "age": anno["age"],
            "race": anno["race"]         
        }

        if self.dataset_id == "mimic":
            text_ = self.mimic_template.format(question=text_)
            target_ = anno["answer"]
        elif self.dataset_id == "mimic-binary":
            text_ = self.mimic_binary_template.format(question=text_)
            target_ = anno["binAnswer"]
        elif self.dataset_id == "mimic-binary-factuality":
            text_ = self.mimic_binary_factuality_template.format(race=race_, gender=gender_, question=text_)
            target_ = anno["binAnswer"]
        elif self.dataset_id == "mimic-open-ended-factuality":
            text_ = self.mimic_binary_factuality_template.format(race=race_, gender=gender_, question=text_)
            target_ = anno["Answer"]

        return ImageTxtSample(image_path=image_path_, text=text_, target=target_, extra=extra_)
    
    # def binQuestion_filter (self):
    #     new_dataset = []
    #     for entry in self.dataset:
    #         if entry["target"] in ["Yes", "No"]:
    #             new_dataset.append(entry)
    #     self.dataset = new_dataset
    #     print(f"filtered for yes/no: {len(self.dataset)} data loaded" )
            
    def __getitem__(self, index: int) -> Tuple[ImageTxtSample, str]:
    
        if self.method_hook:
            item =  self.method_hook.run(self.dataset[index])
        item =  self.dataset[index]
        return item

    def __len__(self) -> int:
        return len(self.dataset)