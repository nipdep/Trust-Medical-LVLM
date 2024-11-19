import logging
import sys 
sys.path.append('../')
# from src.models.test_llava1 import LLaVAChat
from src.models import OpenAIChat, LLaVAChat
#from src.models.unsloth_inference import LLaMaUnSloth
# from src.evaluators.FairnessEval import FairnessEvaluator
from src.grader.metrics import _supported_metrics
from src.datasets.mimic import Mimic, MimicCombinator
from src.tasks.object_base import ObjectBaseTask

from src import ImageTxtSample, _OutputType
from PIL import Image
from io import BytesIO

import os
import json
import pandas as pd
import random
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.special import softmax
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPImageProcessor, CLIPVisionModel
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torch.utils.data import DataLoader
from src.datasets.base import BaseDataset, collate_fn
from src.evaluators import ChatModelEvaluator, ChatModelYesOrNoEvaluator, YesOrNoEvaluator
from src.grader.mimic import OpenEndGrader, BinaryGrader
from typing import List, Dict, Any, Tuple

pid = os.getpid()
print("PID: ", pid)

LOGDIR = "log/"
def saveLog(logName: str, log: pd.DataFrame):
    log.to_csv(LOGDIR + logName + ".csv", index=True)

Race = ["African American", "Asian", "Caucasian", "Hispanic","Native American", "Unknown"]
Gender = ["M", "F", "U"]
dcomb = {"race": Race,
      "gender": Gender}
dataset = MimicCombinator("mimic-factuality", dcomb)
# dataset = Mimic("mimic-factuality")
model =  LLaVAChat(model_id="llava-med", device=torch.device("cuda:2"))
eval_model = OpenAIChat(model_id="gpt-3.5-turbo", device=torch.device("cuda"))

eval = YesOrNoEvaluator(evaluator_id="yes-or-no-fairness", metrics_cfg={})
task = ObjectBaseTask(dataset=dataset, model=model, evaluator=eval, log_file='log/llava_med_mimic_binary_fairness.csv')

result_df = task.pipeline()
result_df = pd.read_csv('log/llava-med_mimic-factuality.csv')
# result_df.head()

grader = BinaryGrader(y_pred=result_df['processed_preds'], y_true=result_df['label'], gender=result_df['gender'], race=result_df['race'])

stat_results = grader.calculate_statistical_parity()
saveLog("race_parityDiff", stat_results['race']['parity_difference_table'])

eo_results = grader.calculate_equal_opportunity()
saveLog("gender_eoDiff",eo_results['gender']['difference_table'])
saveLog("race_eoDiff",eo_results['race']['difference_table'])

eod_results = grader.calculate_equalized_odds()
saveLog("gender_eodDiff", eod_results['gender']["difference_table"])
saveLog("race_eodDiff", eod_results['race']["difference_table"])

oae_results = grader.calculate_overall_accuracy_equality()
saveLog("race_oaeDiff",oae_results['race']['difference_table'])
saveLog("gender_oaeDiff",oae_results['gender']['difference_table'])

te_results = grader.calculate_treatment_equality()
saveLog("race_teDiff",te_results['race']['difference_table'])
saveLog("gender_teDiff",te_results['gender']['difference_table'])

print("Done")