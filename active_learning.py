# Adapted to convenience from https://github.com/castorini/afriberta/blob/6cacc453f3a99a6f902174e8e7f8dd6184c1794f/main.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import os
import shutil
from transformers import pipeline
from source.trainer import TrainingManager
from source.utils import load_config


def save_list(lines, filename):
    data = '\n'.join(str(_).strip() for _ in lines)
    file = open(filename, 'w', encoding="utf-8")
    file.write(data)
    file.close()

def main(args):
    experiment_name = args.experiment_name

    EXPERIMENT_PATH = args.experiment_path
    EXPERIMENT_CONFIG_NAME = args.experiment_config_name

    if not os.path.exists(EXPERIMENT_PATH):
        os.mkdir(EXPERIMENT_PATH)

    experiment_path = os.path.join(EXPERIMENT_PATH, experiment_name)

    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    experiment_config_path = os.path.join(experiment_path, EXPERIMENT_CONFIG_NAME)


    config = load_config(args.config_path)

    langs = ['amh', 'hau', 'lug', 'luo', 'pcm', 'sna', 'tsn', 'wol', 'ewe', 'bam', 'bbj', 'mos', 'zul', 'lin', 'nya', 'twi',
            'fon', 'ibo', 'kin', 'swa', 'xho', 'yor', 'oro']

    dataset = 'dataset/{}_mono.tsv'


    active_learning_steps = args.active_learning_steps
    
    if not os.path.exists(args.data_folder):
        os.mkdir(args.data_folder)

    if not os.path.exists(f'{args.data_folder}/train'):
        os.mkdir(f'{args.data_folder}/train')

    if not os.path.exists(f'{args.data_folder}/eval'):
        os.mkdir(f'{args.data_folder}/eval')

    if not os.path.exists(f'{args.data_folder}/txt'):
        os.mkdir(f'{args.data_folder}/txt')

    for step in range(1, active_learning_steps + 1):
        print('Active Learning Step: {}'.format(step))
        all_evals = []
        # build datasets for the current AL round
        for lang in langs:
            # shuffle the training set for this active learning round
            current_dataset = pd.read_csv(dataset.format(lang), sep='\t')
            current_dataset = current_dataset.sample(frac=1)
            train, test = train_test_split(current_dataset, test_size=0.2, random_state=1234)
            all_evals += test.values.tolist()
            save_list(train.values.tolist(), f'{args.data_folder}/train/train.{lang}')
            save_list(test.values.tolist(), f'{args.data_folder}/eval/eval.{lang}')

        save_list(all_evals, f'{args.data_folder}/eval/all_eval.txt')

        # Creating a data_config dictionary so that we do not have to put it in config.yml 
        data_config = {
            'train': f'{args.data_folder}/train/',
             'eval':{
                'all': f'{args.data_folder}/eval/all_eval.txt', 
                'per_lang': f'{args.data_folder}/eval/'
                        }
                    }

        trainer = TrainingManager(config, experiment_path, step,data_config)
        trainer.train()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser('MLMT_AL')

    parser.add_argument('--experiment_name', type=str, default="active_learning_lm",
        help="Name of experiment. (default: `active_learning_lm`)")
    
    parser.add_argument('--experiment_path', type=str, default='experiments_500ks',
        help='directory path to save all experiments (default: %(default)s)')
    
    parser.add_argument('--experiment_config_name', type=str, default='config.yml',
        help='Config YAML file name (default: %(default)s)')

    parser.add_argument('--active_learning_steps', type=int, default=3,
        help='number of active learning steps (default: %(default)s)')

    parser.add_argument('--data_folder', type=str, default='data',
        help='Folder to save all the datasets generated during the AL experiements (default: %(default)s)')

    parser.add_argument('--config_path', type=str, default='models_configurations/large.yml',
        help='Path to the configuration file for the experiments. (default: %(default)s)')


    args = parser.parse_args() 

    main(args)    
