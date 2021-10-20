"""
Created on Tue Oct 19 18:58:40 2021

@author: cburgheimer
"""
import os
import pandas as pd
import nltk.data

csv_filename = 'data.csv'
txtdir = 'gutenberg/txt'
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

dataframes = []
for txt in os.listdir(txtdir):
    path = os.path.join(txtdir, txt)
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            text = f.read()
        f.close()
        text = text.replace('\n', ' ')
        text_list = tokenizer.tokenize(text)
        dataframe = pd.Series(text_list)
        dataframes.append(dataframe)
df = pd.concat(dataframes)
df.to_csv(csv_filename, encoding = 'utf-8')
