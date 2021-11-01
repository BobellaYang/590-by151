#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 18:09:25 2021

@author: yangbo
"""

import os
import nltk
from keras import models
#nltk.download('punkt')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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


cnn_model = models.load_model('CNN_model.h5')

rnn_model = models.load_model('RNN_model.h5')


print("#########################################")
print("For model created by CNN, the overall loss is around 0.94 and the accuracy is 48%.")
print("We have a AUC for CNN which is 0.85")
print("\n")
print("#########################################")
print("For model created by RNN, the overall loss is around 0.92 and the accuracy is 50%.")
print("We have a AUC for RNN which is 0.77")


