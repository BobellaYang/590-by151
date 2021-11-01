#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 18:09:15 2021

@author: yangbo
"""

import os
import nltk
from keras import models
#nltk.download('punkt')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
from keras import preprocessing
from keras.models import Sequential 
from tensorflow.keras.utils import to_categorical
from keras.layers import Embedding, Flatten, Dense,SimpleRNN, LSTM
from tensorflow.keras import regularizers
from keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from sklearn import metrics

#hyperparameters
maxlen = 100
training_samples = 400
validation_samples = 1000
testing_samples = 200
max_words = 10000
#---------------------------
#USER PARAM
#---------------------------
max_features = 10000    #DEFINES SIZE OF VOCBULARY TO USE
epochs       = 15
batch_size   = 1000
verbose      = 1
embed_dim    = 50        #DIMENSION OF EMBEDING SPACE (SIZE OF VECTOR FOR EACH WORD)
lr           = 0.001    #LEARNING RATE

#read in the data
df = pd.read_csv('clean_data.csv')
texts = list(df.text)
labels = list(df.label)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples] 
y_val = labels[training_samples: training_samples + validation_samples]
x_test = data[training_samples + validation_samples: training_samples + validation_samples + testing_samples]
y_test = labels[training_samples + validation_samples: training_samples + validation_samples + testing_samples]
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
x_val = preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
y_train = to_categorical(y_train, num_classes=3)
y_val = to_categorical(y_val, 3)
y_test = to_categorical(y_test, 3)

print('######################################')
print('Training data',len(x_train), len(y_train))
print('Validation data',len(x_val), len(y_val))
print('Testing data',len(x_test), len(y_test))



#---------------------------
#1D CNN
#---------------------------
print("---------------------------")
print("1D-CNN")  
print("---------------------------")
model = models.Sequential()
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 3, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer=RMSprop(learning_rate=lr), loss='categorical_crossentropy', metrics=['acc']) 

model.summary()
history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
model.save("CNN_model.h5")

#making plots
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("CNN training and validation accuracy")
plt.show()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("CNN training and validation loss")
plt.show()


y_pred_proba = model.predict(x_test)
#calculate AUC of model
auc = metrics.roc_auc_score(y_test, y_pred_proba)
#print AUC score
print(auc)


#Final evaluation
CNN_evaluation = model.evaluate(x_test,y_test)
print('#=================================================')
print('The loss and accuracy for CNN is',CNN_evaluation)




#---------------------------
#RNN
#---------------------------
max_features = 10000    #DEFINES SIZE OF VOCBULARY TO USE
epochs       = 15
batch_size   = 1000
verbose      = 1
embed_dim    = 50        #DIMENSION OF EMBEDING SPACE (SIZE OF VECTOR FOR EACH WORD)
lr           = 0.001    #LEARNING RATE
print("---------------------------")
print("LSTM")  
print("---------------------------")
model = models.Sequential() 
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.LSTM(32, dropout=0.2)) 
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['acc']) 
model.summary()
history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
model.save("RNN_model.h5")

#making plots
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("RNN training and validation accuracy")
plt.show()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("RNN training and validation loss")
plt.show()


y_pred_proba = model.predict(x_test)
#calculate AUC of model
auc = metrics.roc_auc_score(y_test, y_pred_proba)
#print AUC score
print(auc)


#Final evaluation
RNN_evaluation = model.evaluate(x_test,y_test)
print('#=================================================')
print('The loss and accuracy for RNN is',RNN_evaluation)















