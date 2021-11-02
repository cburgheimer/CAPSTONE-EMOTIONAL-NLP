# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 14:40:10 2021

@author: cburgheimer
"""

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

#Encoder-Decoder w/ attention model