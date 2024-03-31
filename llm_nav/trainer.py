from llm_nav.agent import run_navigation
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import logging
import json
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import Trainer
from typing import Dict, Union, Any
import transformers
from arguments import ModelArguments, DataArguments, TrainingArguments
from llm_nav.sim.env import TouchdownBatch
logger = logging.get_logger(__name__)

parser = transformers.HfArgumentParser(
    (ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

split = data_args.dataset.split('/')
split_name = split[1] + '_' + split[2]
eval_env = TouchdownBatch(data_args, splits=[data_args.eval_split], name=split_name)

# The following part will be released after acceptance
