"""
Created on Wed Oct 20 11:42:30 2021

@author: cburgheimer
"""
from dataset_cleaning import *
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
import pandas as pd
'''
label_dict = {'Anticipation':0,'Anger':1, 'Disgust':2,'Fear':3, 'Joy':4, 'Sadness':5, 'Surprise':6, 'Trust':7}
tokenized_data, tokenized_labels, tokenizer, max_len = embed_tokenize_data()'''

'''def measure_performance(y_pred, y_test, THRESHOLD):
    y_test_output = []
    y_test_numpy = np.array(y_test.to_list())
    for i in range(len(label_dict.keys())):
        y_test_output.append(y_test_numpy[:,i])
    
    f1_score_results = []
    for col_idx, col in enumerate(label_dict.keys()):
        print(f'{col} accuracy \n')
        y_pred[col_idx][y_pred[col_idx]>=THRESHOLD] = 1
        y_pred[col_idx][y_pred[col_idx]<THRESHOLD] = 0
        f1_score_results.append(f1_score(y_test_output[col_idx], y_pred[col_idx], average='macro', zero_division=0))
        print(classification_report(y_test_output[col_idx], y_pred[col_idx], zero_division=0))
    print('Total :',np.sum(f1_score_results))'''
    

#Convolutional Neural Network
'''from models.cnn_model import *
run_cnn_model(tokenized_data, tokenized_labels, tokenizer, max_len, label_dict.keys())'''

#Recurrent Neural Network
'''from models.rnn_model import *
run_rnn_model(tokenized_data, tokenized_labels, tokenizer, max_len, label_dict.keys())'''

#Encoder-Decoder w/ attention model


#BERT Fine-Tuned model
'''from models.bert_model import *
dataset = pd.read_csv('data_with_labels/Data_Sentences_W_ Labels.csv',header=0, usecols = ['Sentence', 'Response'])
bert_model, history = run_BERT(dataset)'''

#GPT-2 Fine-Tuned model
from models.GPT_2_model import *
dataset = pd.read_csv('data_with_labels/Data_Sentences_W_ Labels.csv',header=0, usecols = ['Sentence', 'Response'])
gpt_model, gpt_history = run_GPT(dataset)
