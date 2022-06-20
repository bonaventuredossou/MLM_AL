# Adapted to convenience from https://github.com/castorini/afriberta/blob/6cacc453f3a99a6f902174e8e7f8dd6184c1794f/main.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import os
import shutil

from transformers import pipeline
from absl import flags

from source.trainer import TrainingManager
from source.utils import load_config

experiment_name = "active_learning_lm"

EXPERIMENT_PATH = "/content/drive/My Drive/LMs_ALs/experiments_500k"
EXPERIMENT_CONFIG_NAME = "config.yml"

if not os.path.exists(EXPERIMENT_PATH):
    os.mkdir(EXPERIMENT_PATH)

experiment_path = os.path.join(EXPERIMENT_PATH, experiment_name)

if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

experiment_config_path = os.path.join(experiment_path, EXPERIMENT_CONFIG_NAME)

flags.DEFINE_string("config_path", "models_configurations/large.yml", "Config file path")

config = load_config("models_configurations/large.yml")

langs = ['amh', 'hau', 'lug', 'luo', 'pcm', 'sna', 'tsn', 'wol', 'ewe', 'bam', 'bbj', 'mos', 'zul', 'lin', 'nya', 'twi',
         'fon', 'ibo', 'kin', 'swa', 'xho', 'yor', 'oro']

dataset = 'dataset/{}_mono.tsv'


def save_list(lines, filename):
    data = '\n'.join(str(_).strip() for _ in lines)
    file = open(filename, 'w', encoding="utf-8")
    file.write(data)
    file.close()


def main():
    active_learning_steps = 3
    if not os.path.exists('data'):
        os.mkdir('data')

    if not os.path.exists('data/train'):
        os.mkdir('data/train')

    if not os.path.exists('data/eval'):
        os.mkdir('data/eval')

    if not os.path.exists('data/txt'):
        os.mkdir('data/txt')

    for step in range(1, active_learning_steps + 1):
        print('Active Learning Step: {}'.format(step))
        all_evals = []
        # build datasets for the current AL round
        for lang in langs:
            # shuffle the training set for this active learning round
            current_dataset = pd.read_csv(dataset.format(lang), sep='\t')
            current_dataset = current_dataset.sample(frac=1)
            train, test = train_test_split(current_dataset, test_size=0.2, random_state=1234)
            all_evals += test.input.tolist()
            save_list(train.input.tolist(), 'data/train/train.{}'.format(lang))
            save_list(test.input.tolist(), 'data/eval/eval.{}'.format(lang))

        save_list(all_evals, 'data/eval/all_eval.txt')
        trainer = TrainingManager(config, experiment_path, step)
        trainer.train()


if __name__ == '__main__':
    main()
