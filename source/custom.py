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

    # implement the weighted loss idea here
    def compute_loss(self, model, inputs, return_outputs=False):
        torch.cuda.empty_cache()
        labels = inputs.get("labels")

        outputs = model(**inputs)        
        logits = outputs.get("logits")
        logits = logits.view(-1, tokenizer.vocab_size)
        loss = F.cross_entropy(logits, labels.view(-1))
        torch.cuda.empty_cache()
        logger.info('Loss before per-language loss addition: {}'.format(loss))
        
        # next we need to query the model itself on validation samples per language
        eval_dataset_path = Path(held_out_data["eval"]["per_lang"])
        eval_file_paths = eval_dataset_path.glob(EVAL_FILE_PATTERN)
        average_language_loss = 0
        count = 0

        for file_path in eval_file_paths:
            language = file_path.suffix.replace(".", "")

            batch = training_config['per_device_eval_batch_size']
            dataset = EvalDataset(tokenizer, str(file_path))
            number_samples = dataset.__len__()
            
            logger.info('Language: {} - Number of Sample: {}'.format(language, number_samples))
            valid_dataloader = DataLoader(dataset, batch_size=training_config['per_device_eval_batch_size'], shuffle=False)
            total_loss = 0
            data_loader_count = 0
            with torch.no_grad():
                model.eval()            
                for data in valid_dataloader:
                    data['input_ids'] = data['input_ids'].to('cpu') #cuda:1
                    labels_ = data["input_ids"] #.clone()
                    outputs_ = model(**data)
                    del data
                    torch.cuda.empty_cache()
                    logits_ = outputs_.get("logits")
                    logits_ = logits_.view(-1, tokenizer.vocab_size)
                    loss_ = F.cross_entropy(logits_.to('cpu').float(), labels_.view(-1)) # cuda:1
                    # logger.info('Current batch loss: {}'.format(loss_))
                    total_loss += loss_
                    data_loader_count += 1                
                    del labels_
                    del logits_
                    del outputs_
                    torch.cuda.empty_cache()
                language_loss = total_loss/data_loader_count
                average_language_loss += ((1/number_samples) * language_loss)
                logger.info('Loss on Language: {} - {}'.format(language, ((1/number_samples) * language_loss)))
                count += 1
                torch.cuda.empty_cache()
        average_language_loss /= count
        logger.info('Average Loss: {}'.format(average_language_loss))
        loss += average_language_loss
        logger.info('New loss after per-language loss addition: {}'.format(loss))
        torch.cuda.empty_cache()
        model.train()
        return (loss, outputs) if return_outputs else loss


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