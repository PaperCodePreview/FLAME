from dataclasses import dataclass, field
from typing import Literal, List
from typing import Dict, Optional, Sequence
import transformers


@dataclass
class ModelArguments:
    checkpoint_path: Optional[str] = field(default='/home/admin1/Models/touchdown-cot-checkpoint-1000')
    only_attend_immediate_media: bool = field(default=True)

    model_path: Optional[str] = field(default="/home/admin1/Models/OTTER-Image-LLaMA7B-LA-InContext")
    # model_path: Optional[str] = field(default="/home/admin1/Models/Otter-Image-MPT7B")
    precision: Optional[str] = field(default='fp16')
    device: Optional[str] = field(default='cuda')


@dataclass
class DataArguments:
    # legacy_navigation_mode: bool = field(default=False)
    # train_if_data_path: str = field(default='dataset/touchdown_cot/seen/train.json')
    # eval_if_data_path: str = field(default='dataset/touchdown_cot/seen/dev.json')
    # eval_split: str = field(default='dev')
    # dataset: str = field(default='dataset/touchdown/seen')

    # legacy_navigation_mode: bool = field(default=False)
    # train_if_data_path: str = field(default='dataset/map2seq_cot/seen/train.json')
    # eval_if_data_path: str = field(default='dataset/map2seq_cot/seen/dev.json')
    # eval_split: str = field(default='dev')
    # dataset: str = field(default='dataset/map2seq/seen')

    legacy_navigation_mode: bool = field(default=False)
    train_if_data_path: str = field(default='dataset/map2seq_cot_2aux/seen/train.json')
    eval_if_data_path: str = field(default='dataset/map2seq_cot_2aux/seen/dev.json')
    eval_split: str = field(default='dev')
    dataset: str = field(default='dataset/map2seq/seen')

    # legacy_navigation_mode: bool = field(default=False)
    # train_if_data_path: str = field(default='dataset/map2seq_cot_new/seen/train.json')
    # eval_if_data_path: str = field(default='dataset/map2seq_cot_new/seen/dev.json')
    # eval_split: str = field(default='dev')
    # dataset: str = field(default='dataset/map2seq/seen')

    # legacy_navigation_mode: bool = field(default=False)
    # train_if_data_path: str = field(default='dataset/map2seq_cot_new_paradigm/seen/train.json')
    # eval_if_data_path: str = field(default='dataset/map2seq_cot_new_paradigm/seen/dev.json')
    # eval_split: str = field(default='dev')
    # dataset: str = field(default='dataset/map2seq/seen')

    # legacy_navigation_mode: bool = field(default=False)
    # train_if_data_path: str = field(default='dataset/touchdown_cot_new/seen/train.json')
    # eval_if_data_path: str = field(default='dataset/touchdown_cot_new/seen/dev.json')
    # eval_split: str = field(default='dev')
    # dataset: str = field(default='dataset/touchdown/seen')

    # legacy_navigation_mode: bool = field(default=False)
    # train_if_data_path: str = field(default='dataset/touchdown_cot_new_paradigm/seen/train.json')
    # eval_if_data_path: str = field(default='dataset/touchdown_cot_new_paradigm/seen/dev.json')
    # eval_split: str = field(default='dev')
    # dataset: str = field(default='dataset/touchdown/seen')

    img_db: str = field(default='/home/admin1/Datasets/touchdown_feature_final')
    store_feature: bool = field(default=True)
    task: Literal['instruction_following', 'pretraining'] = 'instruction_following'
    train_pre_data_path: str = field(default='dataset/task2_data/train.json')
    eval_pre_data_path: str = field(default='dataset/task2_data/dev.json')
    lazy_preprocess: bool = True
    batch_size: Optional[int] = field(default=64)
    micro_batch_size: Optional[int] = field(default=1)
    env_batch_size: Optional[int] = field(default=1)
    eval_data_size: int = 64
    max_route_len: int = 60

    temperature: float = 0.0
    decoding_paths: int = 1


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # resume_from_checkpoint: Optional[str] = field(default='/home/admin1/FLAME/touchdown_seen_legacy_llama_feature_lr0.0001/checkpoint-2800')
    learning_rate: float = field(default=1e-4)

    output_dir: Optional[str] = field(default='checkpoints')
    optim: str = field(default="adamw_torch")
    # bf16: bool = field(default=True)
    # tf32: bool = field(default=True)
    fp16: bool = field(default=True)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    eval_loss_only: bool = field(default=True)
    warmup_ratio: Optional[int] = field(default=0.01)
    num_train_epochs: Optional[int] = field(default=10)
    save_steps: Optional[int] = field(default=100)
    lr_scheduler_type: Optional[str] = field(default='cosine')
    report_to: Optional[str] = field(default='wandb')
    wandb_project: Optional[str] = field(default='FLAME')
