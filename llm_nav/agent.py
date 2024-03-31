import torch
from collections.abc import Mapping
from torch import nn
from tqdm import tqdm

import copy
from llm_nav.dataset import preprocess
from llm_nav.utils import make_evaluation_data_module

END_SIGNAL = "<|endofchunk|>"
SEP = "\n"

# The following part will be released after acceptance
