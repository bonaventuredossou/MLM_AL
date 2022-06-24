import pandas as pd

path = 'test.tsv'

with open(path, 'r', encoding='utf-8') as file_:
    dataset = file_.readlines()

good_lines = [line.strip('\n').split('\t') for line in dataset if len(line.split('\t')) == 2]

text, label = zip(*good_lines[1:])
# data = pd.read_csv(path, sep='\t')
#
# print(data)

frame = pd.DataFrame()
frame['news_title'] = list(text)
frame['label'] = label

frame.to_csv(path, sep='\t', index=False)
