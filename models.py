"""
Created on Wed Oct 20 11:42:30 2021

@author: cburgheimer
"""
from dataset_cleaning import *
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
from sklearn.model_selection import train_test_split
label_dict = {'Anticipation':0,'Anger':1, 'Disgust':2,'Fear':3, 'Joy':4, 'Sadness':5, 'Surprise':6, 'Trust':7}
tokenized_labels, tokenized_data, tokenizer = embed_tokenize_data()

def measure_performance(y_pred, y_test, THRESHOLD):
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
    print('Total :',np.sum(f1_score_results))
#Bidirectional Neural Network
#Feature Engineering Model

#Convolutional Neural Network

#Recurrent Neural Network
X_train, X_test, y_train, y_test = train_test_split(tokenized_data, tokenized_labels, test_size=0.2, random_state = 42)
vocab_size = len(tokenizer.word_index)+1

main_input = keras.Input(shape=(25,), dtype='int32', name='main_input')
x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=25)(main_input)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv1D(64, 5, activation='relu')(x)
x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)
x = tf.keras.layers.LSTM(512, return_sequences=True)(x)
x = tf.keras.layers.LSTM(512, return_sequences=True)(x)
x = tf.keras.layers.LSTM(512)(x)
x = tf.keras.layers.Dropout(0.3)(x)

output_array = [] 
metrics_array = {}
loss_array = {}
for i, dense_layer in enumerate(label_dict.keys()):
    name = f'binary_output_{i}'
    binary_output = tf.keras.layers.Dense(1, activation='sigmoid', name=name)(x)
    output_array.append(binary_output)
    metrics_array[name] = 'binary_accuracy'
    loss_array[name] = 'binary_crossentropy'

model = tf.keras.models.Model(inputs=main_input, outputs = output_array)
model.compile(optimizer='adadelta',loss=loss_array, metrics=metrics_array)

y_train_output = []
y_train_numpy = np.array(y_train.to_list())
for i in range(len(label_dict.keys())):
    y_train_output.append(y_train_numpy[:,i])

model.fit(X_train, y_train_output, epochs=10, batch_size=512, verbose=0)

THRESHOLD = 0.5

y_pred = model.predict(X_test)
measure_performance(y_pred, y_test, THRESHOLD)

y_pred_train = model.predict(X_train)
measure_performance(y_pred_train, y_train, THRESHOLD)

#Encoder-Decoder w/ attention model

#Endocer-Decoder w/o attention model

#Basic Transformer architecture model

#BERT Fine-Tuned model

#GPT-2 Fine-Tuned model
