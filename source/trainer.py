# Adapted from https://github.com/castorini/afriberta/blob/6cacc453f3a99a6f902174e8e7f8dd6184c1794f/source/trainer.py
import logging
import math
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
import random
import torch
import pandas as pd

import transformers
from transformers import pipeline
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import DataCollatorForWholeWordMask
from transformers import TrainingArguments
from transformers import XLMRobertaConfig
from transformers import XLMRobertaForMaskedLM
from transformers import XLMRobertaTokenizer

from source.custom import CustomTrainer
from source.dataset import EvalDataset
from source.dataset import TrainDataset
from source.utils import create_logger

dataset = '../dataset/{}_mono.tsv'

DEFAULT_XLM_MODEL_SIZE = "xlm-roberta-large"
MLM_PROBABILITY = 0.15
EVAL_FILE_PATTERN = "eval.*"
KEYS_NOT_IN_TRAIN_ARGS = [
    "train_from_scratch",
    "use_whole_word_mask",
    "lang_sampling_factor",
    "resume_training",
]

transformers.logging.set_verbosity_debug()


class TrainingManager:
    """
    A class to manage the training and evaluation of the MLM.

    The actual training is done by a modified version (see custom.py) of the
    huggingface's trainer - https://huggingface.co/transformers/main_classes/trainer.html

    Args:
        config: Loaded configuration from specified yaml file
        experiment_path: path specified to save training outputs
    """

    def __init__(self, config: Dict[str, Any], experiment_path: str) -> None:
        self.data_config = config["data"]
        self.model_config = config["model"]
        self.train_config = config["training"]
        self.train_config["output_dir"] = experiment_path
        self.logger = create_logger(os.path.join(experiment_path, "train_log.txt"))

        # modifying huggingface logger to log into a file
        hf_logger = transformers.logging.get_logger()
        file_handler = logging.FileHandler(os.path.join(experiment_path, "hf_log.txt"))
        file_handler.setLevel(level=logging.DEBUG)
        hf_logger.addHandler(file_handler)

        self.logger.info(f"Experiment Output Path: {experiment_path}")
        self.logger.info(f"Training will be done with this configuration: \n {config} ")

        self._maybe_resume_training()

    def _build_tokenizer(self) -> None:
        """
        Build tokenizer from pretrained sentencepiece model and update config.
        """
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(
            self.model_config.pop("tokenizer_path")
        )
        self.tokenizer.model_max_length = self.model_config["max_length"]

    def _build_model(self) -> None:
        """
        Build model from specified model config.
        """
        self.logger.info("Building model...")
        self._update_model_config()
        xlm_roberta_config = XLMRobertaConfig(**self.model_config)
        self.model = XLMRobertaForMaskedLM(xlm_roberta_config)
        self.logger.info(f"Model built with num parameters: {self.model.num_parameters()}")

    def _build_datasets(self) -> None:
        """
        Build dataset from supplied train and evaluation files.
        """
        self.logger.info("Building datasets...")
        batch_size = self.train_config["per_device_train_batch_size"]
        lang_sampling_factor = self.train_config.pop("lang_sampling_factor")
        self.logger.info(f"Building train dataset from {self.data_config['train']}...")
        self.train_dataset = TrainDataset(
            self.tokenizer,
            self.data_config["train"],
            batch_size,
            self.train_config["output_dir"],
            lang_sampling_factor=lang_sampling_factor,
        )
        self.logger.info(f"No. of training sentences: {len(self.train_dataset)}")
        self.logger.info(f"Building evaluation dataset from {self.data_config['eval']['all']}...")
        self.eval_dataset = EvalDataset(self.tokenizer, self.data_config["eval"]["all"], )
        self.logger.info(f"No. of evaluation sentences: {len(self.eval_dataset)}")

    def train(self, should_generate_first=False) -> None:
        """
        Perform training.
        """
        if should_generate_first:
            self.logger.info("Training stopped. Resuming but generating new samples first")
        else:
            self.logger.info("Starting Training...")
            data_collator = self.collator_class(
                tokenizer=self.tokenizer, mlm_probability=MLM_PROBABILITY
            )

            training_args = TrainingArguments(**self.train_config)
            self.model = self.model.to('cuda')
            # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
            self.trainer = CustomTrainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset 
            )
            train_results = self.trainer.train(model_path=self.model_path)
            train_results_file = os.path.join(self.train_config["output_dir"], "train_results.txt")
            with open(train_results_file, "w") as writer:
                self.logger.info("***** Train results *****")
                for key, value in sorted(train_results.metrics.items()):
                    self.logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            self.logger.info("Training Done! Saving model and model state...")
            self.trainer.save_model()
            self.trainer.state.save_to_json(
                os.path.join(training_args.output_dir, "trainer_state.json")
            )
            self.logger.info("Saving done!")
            self.evaluate()

        eval_dataset_path = Path(self.data_config["eval"]["per_lang"])
        eval_file_paths = eval_dataset_path.glob(EVAL_FILE_PATTERN)
        for file_path in eval_file_paths:
            language = file_path.suffix.replace(".", "")
            self.logger.info('Adding new samples to {}'.format(language))
            new_sentences = self.generate_new_outputs(file_path)
            language_data = pd.read_csv(dataset.format(language), sep='\t')
            updated_language_data = language_data.input.tolist() + new_sentences
            frame = pd.DataFrame()
            frame['input'] = updated_language_data
            frame.to_csv(dataset.format(language), sep='\t', index=False)

    def sample_sequences_from_mlm(self, sequence):
        self.logger.info('Current Input Sequence: {}'.format(sequence))
        self.logger.info('...Sampling from MLM...')
        full_mlm_seqs = []
        unmasker = pipeline("fill-mask", model=self.model_path, tokenizer=self.model_config.pop("tokenizer_path"))
        # choose top-1 option of full sequence
        masked = unmasker(frame)
        generated_sequence = masked[0]['sequence']
        self.logger.info('Generated sequence: {}'.format(generated_sequence))
        return generated_sequence

    def save_list(lines, filename):
        data = '\n'.join(str(_).strip() for _ in lines)
        file = open(filename, 'w', encoding="utf-8")
        file.write(data)
        file.close()

    def generate_new_outputs(self, dataset_path):
        sentences = []
        with open(dataset_path, 'r', encoding='utf-8') as dataset_samples:
            for sentence in dataset_samples.readlines():
                sentences.append(sentence.strip('\n'))

        # we mask the tokens
        sentences_samples_from_mlm = []
        # randomly choosing 1k sentences
        for sentence in random.choices(sentences, k = 1000):
            sentence_split = sentence.strip().split()
            n_tokens = int(len(sentence_split) * MLM_PROBABILITY) + 1

            prompt = sentence_split[:-n_tokens]
            prompt = ' '.join(prompt)
            for _ in range(n_tokens):
                prompt = prompt.strip() + ' <mask>'
                prompt = self.sample_sequences_from_mlm(prompt.strip())
                prompt = prompt.strip()

            sentences_samples_from_mlm.append(prompt)
        
        return sentences_samples_from_mlm



    def evaluate(self) -> None:
        """
        Evaluate trained model on entire evaluation dataset and on per language datasets.
        """
        self.logger.info("Evaluating model...")
        self.logger.info("Evaluating on entire evaluation dataset...")
        self._evaluate()
        self.logger.info("Done! Evaluating on each language...")
        eval_dataset_path = Path(self.data_config["eval"]["per_lang"])
        eval_file_paths = eval_dataset_path.glob(EVAL_FILE_PATTERN)
        for file_path in eval_file_paths:
            language = file_path.suffix.replace(".", "")
            dataset = EvalDataset(self.tokenizer, str(file_path))
            self.logger.info(f"Evaluating {language} with {file_path}...")
            self._evaluate(dataset, language)
        self.logger.info("Completed all evaluations!")

    def _evaluate(self, eval_dataset: Optional[Dataset] = None, language: str = "all") -> None:
        """
        Perform evaluation on a given dataset.
        """
        eval_output = self.trainer.evaluate(eval_dataset)
        eval_output["perplexity"] = math.exp(eval_output["eval_loss"])

        output_eval_file = os.path.join(self.train_config["output_dir"], language + "_eval.txt")
        with open(output_eval_file, "w") as writer:
            self.logger.info(f"***** {language} eval results *****")
            for key, value in sorted(eval_output.items()):
                self.logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

    def _maybe_resume_training(self) -> None:
        """
        Checks if we want to resume the training or not, and launches the appropriate option.
        """
        self._set_data_collator_class()

        if self.train_config.pop("resume_training", None):
            self.model_path = self.train_config["output_dir"]
            self.logger.info(f"Training will resume from {self.model_path}")
            self._build_tokenizer()
            self._build_datasets()
            self._remove_redundant_training_args()
            self.model = XLMRobertaForMaskedLM.from_pretrained(self.model_path)
            self.logger.info(
                f"Model loaded from {self.model_path} with num parameters: {self.model.num_parameters()}"
            )
        else:
            self.model_path = None
            if self.train_config.pop("train_from_scratch"):
                print('Building the model from scratch...')
                self.logger.info("Training from scratch...")
                self._build_tokenizer()
                self._build_model()
            else:
                self.logger.info("Not training from scratch, finetuning pretrained model...")
                self.logger.info("Building tokenizer from pretrained...")
                self.tokenizer = XLMRobertaTokenizer.from_pretrained(DEFAULT_XLM_MODEL_SIZE)
                self.logger.info("Building model from pretrained...")
                self.model = XLMRobertaForMaskedLM.from_pretrained(DEFAULT_XLM_MODEL_SIZE)
            self._build_datasets()

    def _remove_redundant_training_args(self) -> None:
        """
        Removes keys from self.train_config that are not accepted in huggingface's traininng
        arguments.
        """
        for key in KEYS_NOT_IN_TRAIN_ARGS:
            if key in self.train_config:
                del self.train_config[key]

    def _set_data_collator_class(self) -> None:
        """
        Set the data collator class.
        """
        if self.train_config.pop("use_whole_word_mask"):
            self.collator_class = DataCollatorForWholeWordMask
        else:
            self.collator_class = DataCollatorForLanguageModeling

    def _update_model_config(self) -> None:
        """
        Update model configuration.
        """
        self.model_config["vocab_size"] = self.tokenizer.vocab_size
        self.model_config["max_position_embeddings"] = self.model_config["max_length"] + 2
