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
from keras.preprocessing.sequence import pad_sequences
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

def tokenize_label(text):
    label_dict = {'Anticipation':0,'Anger':1, 'Disgust':2,'Fear':3, 'Joy':4, 'Sadness':5, 'Surprise':6, 'Trust':7}
    label = [0,0,0,0,0,0,0,0]
    text = word_tokenize(text)
    for word in text:
        if word in label_dict.keys():
            label[label_dict[word]] = 1
        if word == "None":
            label = [0,0,0,0,0,0,0,0]
    return label
    

def embed_tokenize_data(data_path = 'data_with_labels/Data_Sentences_W_ Labels.csv', filename = 'tokenizer.json'):
    tokenizer = get_tokenizer(filename)
    
    dataset = pd.read_csv(data_path,header=0, usecols = ['Sentence', 'Response'])
    data, labels = dataset.iloc[:,0], dataset.iloc[:,1]
    
    data_cleaned = data.apply(preprocess_text)
    data_cleaned = data_cleaned.apply(word_tokenize)
    max_len = data_cleaned.str.len().max()
    data_cleaned = data_cleaned.apply(' '.join)
    
    tokenized_labels = labels.apply(tokenize_label)
    tokenized_data = tokenizer.texts_to_sequences(data_cleaned)
    tokenized_data = pad_sequences(tokenized_data, padding = 'post')
    
    return tokenized_data, tokenized_labels, tokenizer, max_len


