# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:37:49 2021

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
from tensorflow.keras.optimizers import Adamax
from  tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def decay(epoch):
    if epoch < 3:
        return 5e-3
    elif epoch >= 3 and epoch < 7:
        return 5e-4
    else:
        return 5e-5

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

def create_cnn_model(max_len, vocab_size, labels):
    main_input = tf.keras.Input(shape=(max_len,), dtype='int32', name='main_input')
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len)(main_input)
    x = tf.keras.layers.Conv1D(768, 12, padding='valid', activation='relu')(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    output_array = [] 
    metrics_array = {}
    loss_array = {}
    for i, dense_layer in enumerate(labels):
        output_name = dense_layer
        binary_output = tf.keras.layers.Dense(1, activation='sigmoid', name=output_name)(x)
        output_array.append(binary_output)
        metrics_array[output_name] = 'binary_accuracy'
        loss_array[output_name] = 'binary_crossentropy'
    model = tf.keras.models.Model(inputs=main_input, outputs = output_array)
    optimizer = Adamax(learning_rate=5e-05, epsilon=1e-08, decay=0.01, clipnorm=1.0)
    model.compile(optimizer=optimizer,loss=loss_array, metrics=metrics_array)
    return model

def test_model(model, X_test, y_test, labels):
    evaluation = model.evaluate(X_test, y_test)
    print('Average Loss: ', str(evaluation[0]/8),'\n')
    accuracies = []
    for i, dense_layer in enumerate(labels):
        accuracies.append(evaluation[i+9])
        print(dense_layer, 'Loss: ', str(evaluation[i+1]), '\n')
        print(dense_layer, 'Accuracy: ',str(evaluation[i+9]), '\n')
    print('Average Accuracy:', str(np.sum(accuracies)/8))

def run_cnn_model(tokenized_data, tokenized_labels, tokenizer, max_len, labels):
    X_train, X_test, y_train, y_test, vocab_size = prepare_data(tokenized_data, tokenized_labels, tokenizer, labels)
    model = create_cnn_model(max_len, vocab_size, labels)
    model.summary()
    callbacks = [EarlyStopping(monitor='val_loss', verbose=0, patience=3), TensorBoard(log_dir='logs/cnn_logs'), LearningRateScheduler(decay)]
    history = model.fit(X_train, y_train, validation_split=0.2, batch_size=64, epochs=50, verbose = 0, callbacks=callbacks)
    test_model(model, X_test, y_test, labels)
    return model, history
    