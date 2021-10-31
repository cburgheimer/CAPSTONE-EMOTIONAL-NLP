# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 14:57:03 2021

@author: cburgheimer
"""
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from transformers import TFBertModel,  BertConfig, BertTokenizer
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy


def preprocess_text(text):
    text = re.sub("[^a-zA-Z]", ' ', text)
    text = text.lower().split()
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    return text


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
    data_cleaned = data_cleaned.apply(word_tokenize)
    data_cleaned = data_cleaned.apply(' '.join)
    tokenized_labels = labels.apply(tokenize_label)
    max_len = data_cleaned.str.len().max()
    return data_cleaned, tokenized_labels, max_len

def setup_bert(max_len, model_name = 'bert-base-uncased'):
    config = BertConfig.from_pretrained(model_name)
    config.output_hidden_states = False
    tokenizer = BertTokenizer.from_pretrained(model_name)
    transformer_model = TFBertModel.from_pretrained(model_name, config = config)
    return transformer_model, tokenizer, config

def prepare_data(data_cleaned, tokenized_labels, max_len, tokenizer, label_dict):
    X_train, X_test, y_train, y_test = train_test_split(data_cleaned, tokenized_labels, test_size=0.2, random_state = 42)
    X_train = tokenizer(
        text=X_train.to_list(),
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding=True, 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = False,
        verbose = True)
    X_train = {'input_ids': X_train['input_ids']}
    
    X_test = tokenizer(
        text=X_test.to_list(),
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding=True, 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = False,
        verbose = True)
    X_test = {'input_ids': X_test['input_ids']}
    
    labels = list(label_dict.keys())
    y_train_output = {}
    y_train_numpy = np.array(y_train.to_list())
    for i in range(len(labels)):
        y_train_output[labels[i]] = y_train_numpy[:,i]
    
    y_test_output = {}
    y_test_numpy = np.array(y_test.to_list())
    for i in range(len(labels)):
        y_test_output[labels[i]]= y_test_numpy[:,i]
    
    return X_train, X_test, y_train_output, y_test_output

def build_model(transformer_model, config, max_len, labels):
    BERT = transformer_model.layers[0]
    input_ids = Input(shape=(max_len,), dtype='int32', name='input_ids')
    inputs = {'input_ids': input_ids}
    bert_model = BERT(inputs)[1]
    dropout_layer = Dropout(config.hidden_dropout_prob, name='pooled_outputs')
    pooled_outputs = dropout_layer(bert_model, training=False)
    
    outputs = {}
    metrics_array = {}
    loss_array = {}
    for i, dense_layer in enumerate(labels):
        output_name = f'binary_output_{i}'
        binary_output = Dense(1, activation='sigmoid', kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name=output_name)(pooled_outputs)
        outputs[output_name] = binary_output
        metrics_array[output_name] = BinaryAccuracy()
        loss_array[output_name] = BinaryCrossentropy(from_logits=True)
    print(outputs.keys())
    model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')
    return model, loss_array, metrics_array

def training_model(model, loss, metric, X_train, y_train):
    optimizer = Adam(learning_rate=5e-05, epsilon=1e-08, decay=0.01, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=metric)
    model.fit(X_train, y_train, validation_split=0.2, batch_size=64, epochs=10)
    