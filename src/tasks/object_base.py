from abc import ABC
import pandas as pd
from tqdm import tqdm
from typing import Optional, List, Union, Sequence, Any, Dict, Type
from torch.utils.data import DataLoader
from src.datasets.base import BaseDataset, collate_fn
from src.utils.registry import registry
from src.methods.base import BaseMethod
from src.models.base import BaseChat
from src.evaluators.base import SequentialEvaluator
import warnings
import json
import os
BATCH_SIZE = 128
class ObjectBaseTask(ABC):    
    def __init__(self, dataset: BaseDataset, model: BaseChat, evaluator, method_cfg: Optional[Dict] = {}, dataset_cfg: Optional[Dict] = {}, generation_kwargs: Optional[Dict] = {}, log_file: Optional[str] = None) -> None:
        self.dataset = dataset
        self.model = model
        self.evaluator = evaluator
        self.method_cfg = method_cfg
        self.dataset_cfg = dataset_cfg
        self.generation_kwargs = generation_kwargs
        self.log_file = log_file
        # self.sample_size = sample_size
    
    def get_method(self) -> BaseMethod:
        if not self.method_cfg:
            return None
        
        assert len(self.method_cfg.keys()) == 1
        method_id = list(self.method_cfg.keys())[0]
        method_kwargs = self.method_cfg[method_id]
        method_cls = registry.get_method_class(method_id)
        method = method_cls(method_id, **method_kwargs)
        return method

    def get_dataloader(self, shuffle=True) -> DataLoader:
        dataloader = DataLoader(dataset=self.dataset, batch_size=1, collate_fn=collate_fn, shuffle=shuffle)
        return dataloader

    def eval(self, responses: List[Dict[str, Any]]) -> Dict[str, Union[float, Sequence]]:
        contents: Sequence[str] = [response['content'] for response in responses]
        preds: Sequence[str] = [response['response'] for response in responses]
        labels: Sequence[str] = [response['target'] for response in responses]
        extras: Sequence[str] = [response['extra'] for response in responses]
        token_probs: Sequence[float] = [response['probabilities'] for response in responses]

        # add the token_probs to the extras
        for i in range(len(extras)):
            extras[i]['token_probs'] = token_probs[i]

        results = {}

        result = self.evaluator(preds, labels, extras=extras)
        for key in result.keys():
            if key in results.keys():
                warnings.warn(f"{key} already exists in results.")
        results.update(result)            

        results.update(
            {
                'content': contents if any(contents) else None,
                'pred': preds if any(preds) else None,
                'label': labels if any(labels) else None,
                'extra': extras if any(extras) else None,
            }
        )
        return results

    def save_results(self, results):
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame.from_dict(results)

        # Check if 'extra' column contains dictionaries and expand them if so
        if 'content' in df.columns and df['content'].apply(lambda x: isinstance(x, dict)).any():
            # Normalize the 'extra' column into separate columns and drop the original 'extra' column
            extra_df = pd.json_normalize(df['content'])
            df = df.drop(columns=['content']).join(extra_df)

        # Check if 'extra' column contains dictionaries and expand them if so
        if 'extra' in df.columns and df['extra'].apply(lambda x: isinstance(x, dict)).any():
            # Normalize the 'extra' column into separate columns and drop the original 'extra' column
            extra_df = pd.json_normalize(df['extra'])
            df = df.drop(columns=['extra']).join(extra_df)
                
        # Save the DataFrame to self.log_file, appending if it already exists
        df.to_csv(self.log_file, index=False)
        return df

    def generate(self, dataloader: DataLoader, **generate_kwargs) -> List[Dict[str, Any]]:
        print('len(self.dataset): ', len(dataloader.dataset))
        responses = []
        i = 0
        n = self.dataset_cfg.get('sample_size', len(dataloader.dataset))
  
        for batch_data in tqdm(dataloader, total=n-1):
            for data in batch_data:
                
                message = data['message']
                target = data['target']
                extra: Dict[str, Any] = data['extra']

                response = self.model.chat(messages=message, **generate_kwargs)
                output = {
                    "content": message[0]['content'],
                    "probabilities": response.logprobs,
                    "response": response.content,
                    "target": target,
                    "extra": extra,
                }
            
                responses.append(output)

            if i < n:
                i += 1 
            else:
                break
        
        return responses
        
    def pipeline(self):
        # self.get_handlers()
        dataloader = self.get_dataloader(shuffle=self.dataset_cfg.get('shuffle', True))
        responses = self.generate(dataloader, **self.generation_kwargs)
        results = self.eval(responses)
        result_df = self.save_results(results)
        # scores = self.scorer(results)
        # self.save_results(scores)

        return result_df