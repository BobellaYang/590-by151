#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 20:25:09 2021

@author: yangbo
"""

#packages to use
from keras import layers 
from keras import models
import numpy as np
import warnings
warnings.filterwarnings("ignore")
#get the data that we needed
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt


#define all hyperparameters
CNN = True
DFF_ANN = False
#datasets mnist, mnist_fashion, cifar10
data = "mnist"
savemodel = False

#QUICK INFO ON IMAGE
def get_info(image):
	print("\n------------------------")
	print("INFO")
	print("------------------------")
	print("SHAPE:",image.shape)
	print("MIN:",image.min())
	print("MAX:",image.max())
	print("TYPE:",type(image))
	print("DTYPE:",image.dtype)
#	print(DataFrame(image))




if data == "mnist":
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))
    
    #NORMALIZE
    train_images = train_images.astype('float32') / 255 
    test_images = test_images.astype('float32') / 255  
    
    #DEBUGGING
    NKEEP=60000
    batch_size=int(0.1*NKEEP)
    epochs=10
    print("batch_size",batch_size)
    rand_indices = np.random.permutation(train_images.shape[0])
    train_images=train_images[rand_indices[0:NKEEP],:,:]
    train_labels=train_labels[rand_indices[0:NKEEP]]
    # exit()
        
    #CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
    tmp=train_labels[0]
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    print(tmp, '-->',train_labels[0])
    print("train_labels shape:", train_labels.shape)



elif data == "mnist_fashion":
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))
    
    #NORMALIZE
    train_images = train_images.astype('float32') / 255 
    test_images = test_images.astype('float32') / 255  
    
    #DEBUGGING
    NKEEP=10000
    batch_size=int(0.1*NKEEP)
    epochs=20
    print("batch_size",batch_size)
    rand_indices = np.random.permutation(train_images.shape[0])
    train_images=train_images[rand_indices[0:NKEEP],:,:]
    train_labels=train_labels[rand_indices[0:NKEEP]]
    # exit()
        
    #CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
    tmp=train_labels[0]
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    print(tmp, '-->',train_labels[0])
    print("train_labels shape:", train_labels.shape)


elif data == "cifar10":
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.astype('float32') / 255 
    test_images = test_images.astype('float32') / 255  

#DEBUGGING
    NKEEP=50000
    batch_size=int(0.05*NKEEP)
    epochs=50
    print("batch_size",batch_size)
    rand_indices = np.random.permutation(train_images.shape[0])
    train_images=train_images[rand_indices[0:NKEEP],:,:]
    train_labels=train_labels[rand_indices[0:NKEEP]]
    
#CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
    tmp=train_labels[0]
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    print(tmp, '-->',train_labels[0])
    print("train_labels shape:", train_labels.shape)

get_info(train_images)
get_info(train_labels)

#-------------------------------------
#BUILD MODEL SEQUENTIALLY (LINEAR STACK) FOR CNN
#-------------------------------------

if CNN == True:
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(train_images.shape[1],train_images.shape[2],train_images.shape[3])))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128,(3,3),activation = "relu"))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        model.summary()
        model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
        history = model.fit(train_images, train_labels, epochs = epochs, batch_size = batch_size, validation_split=0.2)
        #-------------------------------------
        #EVALUATE ON TEST DATA
        #-------------------------------------
        train_loss, train_acc = model.evaluate(train_images, train_labels, batch_size=batch_size)
        test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=test_images.shape[0])
        print('train_acc:', train_acc)
        print('test_acc:', test_acc)


#-------------------------------------
#BUILD MODEL FOR DFF_ANN
#-------------------------------------
if DFF_ANN == True:
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=train_images.shape[1]*train_images.shape[2]*train_images.shape[3],))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs = epochs, batch_size = batch_size, validation_split=0.2)
    #-------------------------------------
    #EVALUATE ON TEST DATA
    #-------------------------------------
    train_loss, train_acc = model.evaluate(train_images, train_labels, batch_size=batch_size)
    test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=test_images.shape[0])
    print('train_acc:', train_acc)
    print('test_acc:', test_acc)


#-------------------------------------
#PLOT ACCURACY AND LOSS
#-------------------------------------
plt.plot(history.history['accuracy'], label = "Train accuracy")
plt.plot(history.history['val_accuracy'], label = "Validation accuracy")
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy/loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()


plt.plot(history.history['loss'], label = "Train loss")
plt.plot(history.history['val_loss'], label = "Validation loss")
plt.title('Training and validation loss')
plt.ylabel('Accuracy/loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

#SAVE OUR MODEL
if savemodel:
    model.save("part1_model")






















