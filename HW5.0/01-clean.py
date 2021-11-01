#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 18:08:37 2021

@author: yangbo
"""

import os
import nltk
#nltk.download('punkt')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd


files = os.listdir(os.getcwd()+"/data")


#read in all three txt files with corresponding label
f1 = open(os.getcwd()+"/data/"+files[1])
text1 = f1.read()
f1.close()
file1 = sent_tokenize(text1)


f2 = open(os.getcwd()+"/data/"+ files[2])
text2 = f2.read()
f2.close()
file2 = sent_tokenize(text2)


f3 = open(os.getcwd()+"/data/"+ files[3])
text3 = f3.read()
f3.close()
file3 = sent_tokenize(text3)



texts = []
labels = []

for i in range(len(file1)):
    texts.append(file1[i])
    labels.append(0)
    
for i in range(len(file2)):
    texts.append(file2[i])
    labels.append(1)

for i in range(len(file3)):
    texts.append(file3[i])
    labels.append(2)

#make them into one dataframe
df = pd.DataFrame(columns = ['text','label'])
df.text = texts
df.label = labels


#save it as a csv
df.to_csv('clean_data.csv')


