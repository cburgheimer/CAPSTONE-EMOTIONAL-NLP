"""
Created on Thu Oct 28 14:16:48 2021

@author: cburgheimer
"""
import os
import re
import json
import pandas as pd
import tensorflow as tf
import nltk

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = re.sub("[^a-zA-Z]", ' ', text)
    text = text.lower().split()
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    return text


def create_tokenizer(datadir = 'data', datadir_w_labels = 'data_with_labels'):
    tokenizer = Tokenizer(oov_token = '<OOV>')
    num = 0
    for txt in os.listdir(datadir):
        path = os.path.join(datadir, txt)
        if os.path.isfile(path):
            dataframe = pd.read_csv(path, header=None,)
            dataframe = dataframe.iloc[:,0]
            dataframe = dataframe.apply(preprocess_text)
            dataframe = dataframe.apply(word_tokenize)
            tokenizer.fit_on_texts(dataframe)
            num += 1
            print(num)
            
    print('Done with project gutenberg')
    
    for txt in os.listdir(datadir_w_labels):
        path = os.path.join(datadir_w_labels, txt)
        if os.path.isfile(path):
            dataframe = pd.read_csv(path, header=None,)
            dataframe = dataframe.iloc[:,0]
            dataframe = dataframe.apply(preprocess_text)
            dataframe = dataframe.apply(word_tokenize)
            tokenizer.fit_on_texts(dataframe)
    
    
    json_str = tokenizer.to_json()
    with open('tokenizer.json', 'w', encoding='utf-8') as f:  
        f.write(json.dumps(json_str, ensure_ascii=False))
        f.close()
    
        
if __name__ == '__main__':
      create_tokenizer()