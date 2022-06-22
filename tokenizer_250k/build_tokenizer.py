from pathlib import Path
import sentencepiece as spm

import pandas as pd
from sklearn.model_selection import train_test_split

langs = ['amh', 'hau', 'lug', 'luo', 'pcm', 'sna', 'tsn', 'wol', 'ewe', 'bam', 'bbj', 'mos', 'zul', 'lin', 'nya', 'twi',
         'fon', 'ibo', 'kin', 'swa', 'xho', 'yor', 'oro']

dataset = 'dataset/all_mono/{}_mono.tsv'


def save_list(lines, filename):
    data = '\n'.join(str(_).strip() for _ in lines)
    file = open(filename, 'w', encoding="utf-8")
    file.write(data)
    file.close()


def main():
    all_pds = []
    for lang in langs:
        # shuffle the training set for this active learning round
        current_dataset = pd.read_csv(dataset.format(lang), sep='\t')
        current_dataset = current_dataset.sample(frac=1)
        all_pds.append(current_dataset)

    concats = pd.concat(all_pds)
    save_list(concats.input.tolist(), 'data/txt/all_train.txt')
    spm.SentencePieceTrainer.Train(input='data/txt/all_train.txt', model_prefix='sentencepiece.bpe',
                                   vocab_size=250000, pad_id=1,
                                   unk_id=3,
                                   bos_id=0,
                                   eos_id=2,
                                   pad_piece='<pad>',
                                   unk_piece='<unk>',
                                   bos_piece='<s>',
                                   eos_piece='</s>',
                                   character_coverage=0.9995,
                                   model_type='bpe')


if __name__ == '__main__':
    main()
