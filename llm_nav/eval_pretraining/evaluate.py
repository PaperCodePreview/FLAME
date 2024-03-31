import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 3'
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import transformers
import numpy as np
import torch
from llm_nav.dataset import make_inference_data_module
from llm_nav.eval.models.flamingo import EvalModel
from arguments import ModelArguments, DataArguments, TrainingArguments

from tqdm import tqdm

from llm_nav.eval.eval_model import BaseEvalModel
import argparse


def prepare_eval_samples(test_dataset, num_samples, batch_size, seed):
    np.random.seed(seed)
    # random_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    random_indices = [i for i in range(num_samples)]
    dataset = torch.utils.data.Subset(test_dataset, random_indices)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
    )
    return loader


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def evaluate_captioning(
        args: argparse.Namespace,
        eval_model: BaseEvalModel,
        seed: int = 42,
        min_generation_length: int = 0,
        max_generation_length: int = 500,
        num_beams: int = 1,
        length_penalty: float = -2.0,
):
    data_args.task = 'image_caption'
    test_dataset = make_inference_data_module(eval_model.tokenizer, data_args=data_args)
    test_dataloader = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
        seed,
    )

    results = []
    for batch in tqdm(
            test_dataloader,
            desc=f"Running inference image_caption",
    ):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch["input_ids"],
            batch_first=True,
            padding_value=eval_model.tokenizer.pad_token_id)
        attention_mask = input_ids.ne(eval_model.tokenizer.pad_token_id)
        outputs = eval_model.get_outputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            batch_images=batch["images"],
            min_generation_length=min_generation_length,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )
        results.append(outputs)

    with open('caption_results.json', 'w') as f:
        json.dump(results, f, indent=2)


def evaluate_dense_captioning(
        args: argparse.Namespace,
        eval_model: BaseEvalModel,
        seed: int = 42,
        min_generation_length: int = 0,
        max_generation_length: int = 500,
        num_beams: int = 1,
        length_penalty: float = -2.0,
):
    data_args.task = 'dense_caption'
    test_dataset = make_inference_data_module(eval_model.tokenizer, data_args=data_args)
    test_dataloader = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
        seed,
    )

    results = []
    for batch in tqdm(
            test_dataloader,
            desc=f"Running inference dense_caption",
    ):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch["input_ids"],
            batch_first=True,
            padding_value=eval_model.tokenizer.pad_token_id)
        attention_mask = input_ids.ne(eval_model.tokenizer.pad_token_id)
        outputs = eval_model.get_outputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            batch_images=batch["images"],
            min_generation_length=min_generation_length,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )
        results.append(outputs)

    with open('dense_caption_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # get eval args
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    eval_args = parser.parse_args()
    eval_model = EvalModel(model_args)
    evaluate_captioning(eval_args, eval_model)
