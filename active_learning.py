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

EXPERIMENT_PATH = "experiments"
EXPERIMENT_CONFIG_NAME = "config.yml"


flags.DEFINE_string("config_path", "models_configurations/large.yml", "Config file path")

config = load_config("models_configurations/large.yml")


experiment_path = os.path.join(EXPERIMENT_PATH, experiment_name)
os.makedirs(experiment_path, exist_ok=True)

experiment_config_path = os.path.join(experiment_path, EXPERIMENT_CONFIG_NAME)
# shutil.copy2(FLAGS.config_path, experiment_config_path)


langs = ['amh', 'hau', 'lug', 'luo', 'pcm', 'sna', 'tsn', 'wol', 'ewe', 'bam', 'bbj', 'mos', 'zul', 'lin', 'nya', 'twi',
         'fon', 'ibo', 'kin', 'swa', 'xho', 'yor']

dataset = 'dataset/{}_mono.tsv'

def save_list(lines, filename):
    data = '\n'.join(str(_).strip() for _ in lines)
    file = open(filename, 'w', encoding="utf-8")
    file.write(data)
    file.close()


def main():
    active_learning_steps = 5
    if not os.path.exists('data'):
        os.mkdir('data')

    if not os.path.exists('data/train'):
        os.mkdir('data/train')

    if not os.path.exists('data/eval'):
        os.mkdir('data/eval')

    if not os.path.exists('data/txt'):
        os.mkdir('data/txt')

    for step in range(1, active_learning_steps + 1):
        # this was made to handle a bug at the first iteration
        # the code ccrashed because `pipeline` was not imported.
        # We need to resume training, and generate new samples to continue the training process
        if step != 1:
            all_evals = []
            # build datasets for the current AL round    
            for lang in langs:
                # shuffle the training set for this active learning round
                current_dataset = pd.read_csv(dataset.format(lang), sep='\t')
                current_dataset = current_dataset.sample(frac=1)
                # if lang == 'yor':
                    # current_dataset = current_dataset.sample(n = 65000, random_state=1234) # the yoruba dataset is huge so we downsample it for computational reason
                train, test = train_test_split(current_dataset, test_size=0.2, random_state=1234)            
                all_evals += test.input.tolist()
                save_list(train.input.tolist(), 'data/train/train.{}'.format(lang))
                save_list(test.input.tolist(), 'data/eval/eval.{}'.format(lang))
            save_list(all_evals, 'data/eval/all_eval.txt'.format(lang))
            config["data"]["generate_first"] = False
        else:
            config["data"]["generate_first"] = True
        trainer = TrainingManager(config, experiment_path, step)
        trainer.train()


if __name__ == '__main__':
    main()
