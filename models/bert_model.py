# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 14:57:03 2021

@author: cburgheimer
"""
import pandas as pd
import numpy as np
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from transformers import TFBertModel,  BertConfig, BertTokenizer
from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from  tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard


def preprocess_text(text):
    text = re.sub("[^a-zA-Z]", ' ', text)
    text = text.lower().split()
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

def decay(epoch):
    if epoch < 3:
        return 5e-3
    elif epoch >= 3 and epoch < 7:
        return 5e-4
    else:
        return 5e-5

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

def clean_data(dataset):
    data, labels = dataset['Sentence'], dataset['Response']
    data_cleaned = data.apply(preprocess_text)
    
    empty = data_cleaned[data_cleaned==''].index.to_list()
    data_cleaned.drop(empty, inplace = True)
    data_cleaned.reset_index(drop=True, inplace=True)
    labels.drop(empty, inplace = True)
    labels.reset_index(drop=True, inplace=True)
    
    max_len = data_cleaned.str.split().str.len().max()
    tokenized_labels = labels.apply(tokenize_label)
    data_cleaned = data_cleaned.str.split()
    return data_cleaned, tokenized_labels, max_len

def setup_bert(max_len, model_name = 'bert-base-uncased'):
    config = BertConfig.from_pretrained(model_name)
    config.output_hidden_states = False
    tokenizer = BertTokenizer.from_pretrained(model_name, config = config)
    transformer_model = TFBertModel.from_pretrained(model_name, config = config)
    return transformer_model, tokenizer, config

def prepare_data(data_cleaned, tokenized_labels, max_len, tokenizer, labels):  
    encodings =  tokenizer(data_cleaned.to_list(), max_length=max_len, is_split_into_words=True, padding = True, truncation = True, verbose=True)
    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    X_train, X_test, y_train, y_test, attention_train, attention_test = train_test_split(input_ids, tokenized_labels, attention_masks, test_size=0.2, random_state = 42)
    X_train = {'input_ids': np.array(X_train), 'attention_mask': np.array(attention_train)}
    X_test = {'input_ids': np.array(X_test), 'attention_mask': np.array(attention_test)}
    y_train_output = {}
    y_train_numpy = np.array(y_train.to_list())
    for i in range(len(labels)):
        y_train_output[labels[i]] = y_train_numpy[:,i]
    y_train = y_train_output
    
    y_test_output = {}
    y_test_numpy = np.array(y_test.to_list())
    for i in range(len(labels)):
        y_test_output[labels[i]]= y_test_numpy[:,i]
    y_test = y_test_output
    return X_train, X_test, y_train, y_test

def build_model(transformer_model, config, max_len, labels):
    BERT = transformer_model.layers[0]
    input_ids = Input(shape=(max_len,), dtype='int32', name='input_ids')
    attention_mask = Input(shape=(max_len,), name='attention_mask', dtype='int32')
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    bert_model = BERT(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'])[0]
    p_bert_model = GlobalAveragePooling1D()(bert_model)
    dropout_layer = Dropout(config.hidden_dropout_prob, name='pooled_outputs')
    pooled_outputs = dropout_layer(p_bert_model, training=False)
    
    outputs = []
    metrics_array = {}
    loss_array = {}
    for i, dense_layer in enumerate(labels):
        output_name = dense_layer
        binary_output = Dense(1, activation='sigmoid', kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name=output_name)(pooled_outputs)
        outputs.append(binary_output)
        metrics_array[output_name] = BinaryAccuracy()
        loss_array[output_name] = BinaryCrossentropy()
    model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')
    return model, loss_array, metrics_array

def training_model(model, loss, metric, X_train, y_train):
    optimizer = Adamax(learning_rate=5e-05, epsilon=1e-08, decay=0.01, clipnorm=1.0)
    callbacks = [EarlyStopping(monitor='val_loss', verbose=0, patience=3), TensorBoard(log_dir='/logs/bert_logs'), LearningRateScheduler(decay)]
    model.compile(optimizer=optimizer, loss=loss, metrics=metric)
    history = model.fit(X_train, y_train, validation_split=0.2, batch_size=64, epochs=50, verbose=0, callbacks = callbacks)
    return model, history

    
def test_model(model, X_test, y_test, labels):
    evaluation = model.evaluate(X_test, y_test)
    print('Average Loss: ', str(evaluation[0]/8),'\n')
    accuracies = []
    for i, dense_layer in enumerate(labels):
        accuracies.append(evaluation[i+9])
        print(dense_layer, 'Loss: ', str(evaluation[i+1]), '\n')
        print(dense_layer, 'Accuracy: ',str(evaluation[i+9]), '\n')
    print('Average Accuracy:', str(np.sum(accuracies)/8))
    
def run_BERT(dataset, model_name = None):
    label_dict = {'Anticipation':0,'Anger':1, 'Disgust':2,'Fear':3, 'Joy':4, 'Sadness':5, 'Surprise':6, 'Trust':7}
    labels = list(label_dict.keys())
    
    data_cleaned, tokenized_labels, max_len = clean_data(dataset)
    if model_name is not None:
        transformer_model, tokenizer, config = setup_bert(max_len, model_name)
    else:
        transformer_model, tokenizer, config = setup_bert(max_len)
    X_train, X_test, y_train, y_test = prepare_data(data_cleaned, tokenized_labels, max_len, tokenizer, labels)
    model, loss_array, metrics_array = build_model(transformer_model, config, max_len, labels)
    model.summary()
    model, history = training_model(model, loss_array, metrics_array, X_train, y_train)
    test_model(model, X_test, y_test, labels)
    return model, history