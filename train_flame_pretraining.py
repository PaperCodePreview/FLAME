import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 3'
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import transformers

from llm_nav.trainer import FlameTrainer, model_args, data_args, training_args

from llm_nav.config import FlamingoConfig
from llm_nav.model.modeling_flamingo import FlamingoForConditionalGeneration
from llm_nav.dataset import make_supervised_data_module


def train():
    gradient_accumulation_steps = data_args.batch_size // data_args.micro_batch_size
    training_args.gradient_accumulation_steps = gradient_accumulation_steps
    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    if training_args.resume_from_checkpoint:
        model_args.model_path = training_args.resume_from_checkpoint
    model_config = FlamingoConfig.from_pretrained(model_args.model_path)
    model_config.only_attend_immediate_media = model_args.only_attend_immediate_media
    model_config.feature_as_input = data_args.store_feature
    model = FlamingoForConditionalGeneration.from_pretrained(
        model_args.model_path,
        config=model_config,
        device_map="auto"
    )
    model.config.use_cache = False

    tokenizer = model.text_tokenizer
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    data_args.batch_size = training_args.per_device_train_batch_size
    split = data_args.dataset.split('/')
    split_name = split[1] + '_' + split[2]
    model_name = "llama" if "llama" in model_args.model_path.lower() else "mpt"
    if_mode = "legacy" if data_args.legacy_navigation_mode else "cot"
    media_mode = "feature" if data_args.store_feature is False else "image"
    task_name = data_args.task if data_args.task != 'instruction_following' else f'{split_name}_{if_mode}'

    final_training_args = transformers.TrainingArguments(
        remove_unused_columns=False,
        per_device_train_batch_size=data_args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        output_dir=f'{task_name}_{model_name}_{media_mode}_lr{training_args.learning_rate}_no_brackets_stride2',
        warmup_ratio=training_args.warmup_ratio,
        num_train_epochs=5,
        learning_rate=1e-5,
        # tf32=training_args.tf32,
        optim=training_args.optim,
        lr_scheduler_type=training_args.lr_scheduler_type,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        save_total_limit=2,
        save_steps=training_args.save_steps,
        report_to=training_args.report_to
    )

    trainer = FlameTrainer(model=model,
                           tokenizer=tokenizer,
                           args=final_training_args,
                           **data_module)

    # trainer.evaluate()
    trainer.train()
    trainer._save_checkpoint(model, trial=None, metrics=None)


if __name__ == "__main__":
    train()
