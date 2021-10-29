"""
Created on Thu Oct 28 12:39:17 2021

@author: cburgheimer
"""
import pandas as pd
import nltk
nltk.download()

data_path = 'data_with_labels/Data_Sentences_W_ Labels.csv'

dataset_with_labels = pd.read_csv(data_path,header=0, usecols = ['Sentence', 'Response'])

training_data = dataset_with_labels.sample(frac=0.8,random_state=42)
testing_data = dataset_with_labels.drop(training_data.index)

train_X, train_y = training_data.iloc[:,0], training_data.iloc[:,1]
test_X, test_y = testing_data.iloc[:,0], testing_data.iloc[:,1]

