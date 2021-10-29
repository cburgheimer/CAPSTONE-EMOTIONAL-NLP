"""
Created on Thu Oct 28 12:39:17 2021

@author: cburgheimer
"""
import os
import re
import json
import pandas as pd
import tensorflow as tf
import nltk
import json

from tokenizer_prep import preprocess_text
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def get_tokenizer(filename = 'tokenizer.json'):
    with open(filename) as f: 
        tokenizer_json = json.load(f) 
        tokenizer = tokenizer_from_json(tokenizer_json)
        f.close()
    return tokenizer



if __name__ == '__main__':
    data_path = 'data_with_labels/Data_Sentences_W_ Labels.csv'
    
    tokenizer = get_tokenizer()
    
    dataset_with_labels = pd.read_csv(data_path,header=0, usecols = ['Sentence', 'Response'])
    data, labels = dataset_with_labels.iloc[:,0], dataset_with_labels.iloc[:,1]
    
    data = data.apply(preprocess_text)
    data = data.apply(word_tokenize)
    tokenizer.fit_on_texts(data)
    


