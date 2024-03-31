import os

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

import transformers
import json

from llm_nav.sim.env import TouchdownBatch
from arguments import ModelArguments, DataArguments
from llm_nav.agent import run_navigation
from llm_nav.config import FlamingoConfig
from llm_nav.model.modeling_flamingo import FlamingoForConditionalGeneration

parser = transformers.HfArgumentParser(
    (ModelArguments, DataArguments))
model_args, data_args = parser.parse_args_into_dataclasses()

if __name__ == "__main__":
    model_config = FlamingoConfig.from_pretrained(model_args.checkpoint_path)
    model_config.only_attend_immediate_media = model_args.only_attend_immediate_media
    model_config.feature_as_input = data_args.store_feature
    model = FlamingoForConditionalGeneration.from_pretrained(
        model_args.checkpoint_path,
        torch_dtype=torch.float16,
        config=model_config,
        device_map="auto"
    )
    model.eval()
    tokenizer = model.text_tokenizer
    split = data_args.dataset.split('/')
    split_name = split[1] + '_' + split[2]
    # data_args.eval_data_size = -1
    eval_env = TouchdownBatch(data_args, splits=[data_args.eval_split], name=split_name)
    metrics, trajs_record = run_navigation(eval_env, model, tokenizer, data_args)
    print(metrics)

    with open(f"{model_args.checkpoint_path}/nav_result_{data_args.eval_split}.json", "w") as f:
        json.dump({"metrics": metrics, "trajs": trajs_record}, f, indent=2)
