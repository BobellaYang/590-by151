#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 02:06:58 2021

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


#---------------------------
#USER PARAM
#---------------------------
epochs       = 10
batch_size   = 100
n_bottleneck = 10
learn_rate   = 0.01    #LEARNING RATE


(X, Y), (test_images, test_labels) = mnist.load_data()

#Normalize
X = X.astype('float32') / 255.

#inject noise
X2 = X + 1*np.random.uniform(0,1,X.shape)

#reshape
X = X.reshape(60000,28,28,1)
X2 = X2.reshape(60000,28,28,1)

input_img = Input(shape=(28, 28, 1))

x = Conv2D(10, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(6, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((4, 4), padding='same')(x)

x = Conv2D(6, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((4, 4))(x)
x = Conv2D(10, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='linear', padding='same')(x)

model = Model(input_img, decoded)

model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.summary()

history = model.fit(X,X, epochs = epochs, batch_size = batch_size,validation_split=0.2)

model.save("model6.2.h5")

#making plots
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("Conv AE training and validation loss")
plt.show()

#Fashion data
(X_fashion, Y_fashion), (test_images_fashion, test_labels_fashion) = fashion_mnist.load_data()

#Normalize
X_fashion = X_fashion/np.max(X_fashion)


#reshape
X_fashion = X_fashion.reshape(60000,28,28,1)
test_images_fashion = test_images_fashion.reshape(10000,28,28,1)
fig = plt.figure()
import random
for i in range(8):
    ints = random.randint(0, 59999)
    plt.subplot(4,4,i+1)
    plt.imshow(X[ints].reshape(28,28))

fig


fig.savefig('HW6.2-history.png')

#Counting anomaly
# Get train MAE loss.
x_train_pred = model.predict(test_images.reshape(10000,28,28,1))
train_mae_loss = np.mean(np.abs(x_train_pred - test_images.reshape(10000,28,28,1)), axis=1)


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

anomalies_an = test_mae_loss > threshold

print("Number of anomaly for anomly dataset is: ", np.sum(anomalies_an))



