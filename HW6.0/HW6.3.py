#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 03:19:42 2021

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
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D, UpSampling2D
from sklearn import metrics
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras import Model
from keras.datasets import cifar10, cifar100

#---------------------------
#USER PARAM
#---------------------------
epochs       = 10
batch_size   = 100
n_bottleneck = 10
learn_rate   = 0.01    #LEARNING RATE


(X, Y), (test_images, test_labels) = cifar10.load_data()

#Normalize
X = X.astype('float32') / 255.

#inject noise
X2 = X + 1*np.random.uniform(0,1,X.shape)

#reshape
X = X.reshape(50000,32,32,3)


input_img = Input(shape=(32,32,3))

x = Conv2D(12, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (4, 4), activation='relu', padding='same')(x)
encoded = MaxPooling2D((6, 6), padding='same')(x)

x = Conv2D(8, (4, 4), activation='relu', padding='same')(encoded)
x = UpSampling2D((6, 6))(x)
x = Conv2D(12, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='linear', padding='same')(x)

model = Model(input_img, decoded)

model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.summary()

history = model.fit(X,X, epochs = epochs, batch_size = batch_size,validation_split=0.2)

model.save("model6.3.h5")

#making plots
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("Conv AE training and validation loss for cifar")
plt.show()

#Fashion data
(X_fashion, Y_fashion), (test_images_fashion, test_labels_fashion) = cifar100.load_data()

#Normalize
X_fashion = X_fashion/np.max(X_fashion)


#reshape
X_fashion = X_fashion.reshape(50000,32,32,3)
test_images_fashion = test_images_fashion.reshape(10000,32,32,3)
fig = plt.figure()
import random
for i in range(6):
    ints = random.randint(0, 49999)
    plt.subplot(3,3,i+1)
    plt.imshow(X[ints].reshape(32,32,3))

fig


fig.savefig('HW6.3-history.png')

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
x_test_pred = model.predict(test_images_fashion)
test_mae_loss = np.mean(np.abs(x_test_pred - test_images_fashion), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))
error = model.evaluate(X_fashion,X_fashion)

anomalies_an = error > threshold

print("Number of anomaly for anomly dataset is: ", np.sum(anomalies_an))



