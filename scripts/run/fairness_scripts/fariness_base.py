import os

import torch
from argparse import ArgumentParser

from src.models import OpenAIChat, LLaVAChat
from src.datasets.mimic import Mimic, MimicCombinator
from src.tasks.object_base import ObjectBaseTask
from src.evaluators import ChatModelEvaluator, ChatModelYesOrNoEvaluator, YesOrNoEvaluator
from src.grader.mimic import OpenEndGrader, BinaryGrader

# write the argparser
parser = ArgumentParser()
parser.add_argument("--model_id", type=str, default="llava-med")
parser.add_argument("--dataset_id", type=str, default="mimic-factuality")
parser.add_argument("--save_path", type=str, default="./log/temp.csv")
parser.add_argument("--grader", true_action="store_true", default=False)

if __name__ == "__main__":
    args = parser.parse_args()
    binary = False
    factuality = False

    # load dataset
    dataset_id = args.dataset_id
    dataset_split_list = dataset_id.split('-')
    dataset = dataset_split_list[0]
    if len(dataset_split_list) == 2:
        if dataset_split_list[1] == "factuality":
            factuality = True
        elif dataset_split_list[1] == "binary":
            binary = True
    elif len(dataset_split_list) == 3:
            factuality = True
            binary = True

    if dataset == "mimic":
        dataset = Mimic(args.dataset_id)
    else:
        dataset = MimicCombinator(args.dataset_id)

    # load model 
    model_id = args.model_id
    model_name = model_id.split('-')[0]
    if model_name == "llava":
        model = LLaVAChat(model_id=args.model_id, device=torch.device("cuda"))
    else:
        Exception("Model not found under the given model_id", model_id)
    
    # load evaluator
    if binary:
        eval = YesOrNoEvaluator(evaluator_id="yes-or-no-fairness", metrics_cfg={})
    else:
        similarity_prompt = "give the {label} and {pred}, check whether they are similar or not, Answer should be Yes or No"
        eval_model = OpenAIChat(model_id="gpt-3.5-turbo")
        eval = ChatModelYesOrNoEvaluator(evaluator_id="fairness", chatmodel=eval_model, prompt_template=similarity_prompt, generation_kwargs={}, metrics_cfg={})

    
    task = ObjectBaseTask(dataset=dataset, model=model, evaluator=eval, log_file=args.save_path)
    result_df = task.pipeline()

