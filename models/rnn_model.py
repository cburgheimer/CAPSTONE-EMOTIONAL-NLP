# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 14:40:10 2021

@author: cburgheimer
"""
import os
import re
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
import json

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def prepare_data(tokenized_data, tokenized_labels, tokenizer, labels):
    X_train, X_test, y_train, y_test = train_test_split(tokenized_data, tokenized_labels, test_size=0.2, random_state = 42)
    vocab_size = len(tokenizer.word_index)+1
    
    y_train_numpy = np.array(y_train.to_list())
    y_train = []
    for i in range(len(labels)):
        y_train.append(y_train_numpy[:,i])
        
    y_test_numpy = np.array(y_test.to_list())
    y_test = []
    for i in range(len(labels)):
        y_test.append(y_test_numpy[:,i])
        
    return X_train, X_test, y_train, y_test, vocab_size
    
def create_model(max_len, vocab_size, labels):
    main_input = tf.keras.Input(shape=(max_len,), dtype='int32', name='main_input')
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=25)(main_input)
    x = tf.keras.layers.GRU(1024, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=8)(x)
    x = tf.keras.layers.LSTM(1024, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(1024, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(1024)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    output_array = [] 
    metrics_array = {}
    loss_array = {}
    for i, dense_layer in enumerate(labels):
        name = f'binary_output_{i}'
        binary_output = tf.keras.layers.Dense(1, activation='sigmoid', name=name)(x)
        output_array.append(binary_output)
        metrics_array[name] = 'binary_accuracy'
        loss_array[name] = 'binary_crossentropy'
    model = tf.keras.models.Model(inputs=main_input, outputs = output_array)
    model.compile(optimizer='adadelta',loss=loss_array, metrics=metrics_array)
    return model

def run_model(tokenized_data, tokenized_labels, tokenizer, max_len, labels):
    X_train, X_test, y_train, y_test, vocab_size = prepare_data(tokenized_data, tokenized_labels, tokenizer, labels)
    model = create_model(max_len, vocab_size, labels)
    model.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=10)
    model.evaluate(X_test, y_test)