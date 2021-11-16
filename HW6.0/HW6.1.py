#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 00:57:03 2021

@author: yangbo
"""

import os
from keras import models
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
from keras.datasets import mnist
from keras.datasets import fashion_mnist


#---------------------------
#USER PARAM
#---------------------------
epochs       = 15
batch_size   = 1000
n_bottleneck = 50
learn_rate   = 0.01    #LEARNING RATE


(X, Y), (test_images, test_labels) = mnist.load_data()

#Normalize
X = X/np.max(X)

#inject noise
X2 = X + 1*np.random.uniform(0,1,X.shape)

#reshape
X = X.reshape(60000,28*28)
X2 = X2.reshape(60000,28*28)

model = models.Sequential()
model.add(layers.Dense(n_bottleneck, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(28*28, activation='relu'))

model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.summary()

history = model.fit(X,X, epochs = epochs, batch_size = batch_size,validation_split=0.2)

model.save("model6.1.h5")

#making plots
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("AE training and validation loss")
plt.show()

#Fashion data
(X_fashion, Y_fashion), (test_images_fashion, test_labels_fashion) = fashion_mnist.load_data()

#Normalize
X_fashion = X_fashion/np.max(X_fashion)


#reshape
X_fashion = X_fashion.reshape(60000,28*28)

fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.imshow(X[i].reshape(28,28))

fig


fig.savefig('HW6.1-history.png')

#Counting anomaly
# Get train MAE loss.
x_train_pred = model.predict(X)
train_mae_loss = np.mean(np.abs(x_train_pred - X), axis=1)


# Get reconstruction loss threshold.
threshold = 4*model.evaluate(X,X,batch_size = batch_size)
print("Reconstruction error threshold: ", threshold)

e = model.evaluate(X,X,batch_size = batch_size)
anomalies = e > threshold

print("Number of anomaly for train dataset is: ", np.sum(anomalies))

# Get test MAE loss.
x_test_pred = model.predict(X_fashion)
test_mae_loss = np.mean(np.abs(x_test_pred - X_fashion), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

anomalies = test_mae_loss > threshold

print("Number of anomaly for anomly dataset is: ", np.sum(anomalies))



