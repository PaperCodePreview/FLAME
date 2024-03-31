import copy
import os
import random
from dataclasses import dataclass, field
import json
import logging
from typing import Dict, Optional, Sequence, List, Any, Union
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
import transformers
from transformers import CLIPImageProcessor
from torch.utils.data import Dataset
import lmdb
import cv2
from PIL import Image
from functools import lru_cache

from llm_nav import conversation as conversation_lib

IGNORE_INDEX = -100

# The following part will be released after acceptance
