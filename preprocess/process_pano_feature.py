import os
import argparse
import lmdb
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import pickle

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

import torch

from transformers import CLIPImageProcessor
from flamingo.modeling_flamingo import FlamingoForConditionalGeneration


# The following part will be released after acceptance
