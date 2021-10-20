"""
Created on Tue Oct 19 18:58:40 2021

@author: cburgheimer
"""
import os
import pandas as pd
import nltk
nltk.download('punkt')

txtdir = 'gutenberg/txt'
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

num=0
for txt in os.listdir(txtdir):
    num += 1
    path = os.path.join(txtdir, txt)
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            text = f.read()
        f.close()
        text = text.replace('\n', ' ')
        text_list = tokenizer.tokenize(text)
        dataframe = pd.Series(text_list)
        csv_filename = 'data/data_'+str(num)+'.csv'
        dataframe.to_csv(csv_filename, index = False, encoding = 'utf-8')
