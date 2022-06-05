# Adapted from https://github.com/castorini/afriberta/blob/6cacc453f3a99a6f902174e8e7f8dd6184c1794f/src/custom.py

import math
import random
from typing import Dict
from typing import Tuple
import torch.nn as nn

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, pipeline
from source.dataset import EvalDataset
from pathlib import Path
import torch.nn.functional as F
import gc
import os

from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import DataCollatorForWholeWordMask
from transformers import TrainingArguments
from transformers import XLMRobertaConfig
from transformers import XLMRobertaForMaskedLM
from transformers import XLMRobertaTokenizer
from source.utils import load_config
from source.utils import create_logger

config = load_config("models_configurations/large.yml")

held_out_data = config["data"]
model_config = config["model"]
training_config = config["training"]
tokenizer = XLMRobertaTokenizer.from_pretrained(model_config.pop("tokenizer_path"))
tokenizer.model_max_length = model_config["max_length"]
EVAL_FILE_PATTERN = "eval.*"

experiment_path = "/home/mila/b/bonaventure.dossou/emnlp22/experiments/"
logger = create_logger(os.path.join(experiment_path, "train_results.txt"))

class CustomTrainer(Trainer):
    def __init__(self, **kwargs) -> None:
        super(CustomTrainer, self).__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Overwrites original method to use a worker init function.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            worker_init_fn=self.worker_init_fn,
        )

    @staticmethod
    def get_worker_shard(
        examples: Dict[str, np.ndarray], num_workers: int, worker_id: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """
        for each language in the dataset, divide the language dataset into approx num_workers shards
        and retrieve corresponding shard for the worker using its ID.
        """
        shard = {}
        shard_stats = {}
        for language, inputs in examples.items():
            num_examples_per_worker = math.ceil(len(inputs) / num_workers)
            begin_index, end_index = (
                num_examples_per_worker * worker_id,
                num_examples_per_worker * (worker_id + 1),
            )
            shard[language] = inputs[begin_index:end_index]
            shard_stats[language] = len(shard[language])
        shard_stats["total"] = sum(shard_stats.values())
        return shard, shard_stats

    def worker_init_fn(self, worker_id: int) -> None:
        """
        worker init function to change random state per worker.
        """
        np.random.seed(np.random.get_state()[1][0] + worker_id + random.randint(1, 1000))

        worker_info = torch.utils.data.get_worker_info()
        worker_info.dataset.set_worker_id(worker_id)
        worker_info.dataset.examples, shard_stats = self.get_worker_shard(
            worker_info.dataset.examples, worker_info.num_workers, worker_id
        )
        worker_info.dataset.logger.info(
            f"Stats for shard created for worker {worker_id}: \n {shard_stats}"
        )
        worker_info.dataset.create_language_index_mapping()