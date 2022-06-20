import sentencepiece as spm
import pandas as pd

langs = ['amh', 'hau', 'lug', 'luo', 'pcm', 'sna', 'tsn', 'wol', 'ewe', 'bam', 'bbj', 'mos', 'zul', 'lin', 'nya', 'twi',
         'fon', 'ibo', 'kin', 'swa', 'xho', 'yor', 'oro']

dataset = '../dataset/{}_mono.tsv'


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
    save_list(concats.input.tolist(), '../data/txt/all_train.txt')
    print('Building SPM')
    spm.SentencePieceTrainer.Train(input='../data/txt/all_train.txt', model_prefix='sentencepiece.bpe',
                                   vocab_size=512000,
                                   character_coverage=0.9995, model_type='bpe')


if __name__ == '__main__':
    main()
