#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 21:41:18 2021

@author: yangbo
"""

import statistics
from statistics import mean
import pandas as pd
from sklearn.model_selection import train_test_split  
import numpy as np
from scipy.optimize import fmin, minimize
import matplotlib.pyplot as plt

class Data:
    def __init__(self, filename):
        self.data = pd.read_json(filename)
        self.age = pd.read_json(filename)['x']
        self.weight = pd.read_json(filename)['y']
        self.is_adult = pd.read_json(filename)['is_adult']

    def normalization(self, x, y):
        newx = [(i - np.mean(x))/np.std(x) for i in x]
        newy = [(i - np.mean(y))/np.std(y) for i in y]
        return newx, newy

    def partition_lineardata(self):
        #partition for linear data, only keep age less than 18 data points
        age_18 = self.age <= 18
        x = self.age[age_18]
        y = self.weight[age_18]

        x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size = 0.8)
        x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size = 0.5)

        return x_train, x_valid, x_test, y_train, y_valid, y_test

    def loss_linear(self, p):
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.partition_lineardata()
        nx,ny = self.normalization(x_train,y_train)
        y = np.dot(nx, p[0]) + p[1]
        total = ny - y
        a = np.power(total, 2)
        MSE = np.sqrt(np.sum(a))/len(ny)
        return MSE    
    
    def optimize(self):
         #TRAIN MODEL USING SCIPY OPTIMIZER
         po=np.random.uniform(0.5,1.,size=2)
         res = minimize(self.loss_linear, po, method='Nelder-Mead')
         return res.x
    
    def plot_linear(self):
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.partition_lineardata()
        fig, ax = plt.subplots()
        ax.plot(x_train, y_train, 'o', label='Training Set')
        ax.plot(x_valid, y_valid, 'o', label='Validation Set')
        ax.plot(x_test, y_test, 'o', label='Testing Set')
        p = self.optimize()
        xn, yn = self.normalization(x_train,y_train)
        #the optimal value I got was below, therefore, I put the value down
        y_pred = np.dot(xn, p[0]) + p[1]
        ax.plot(x_train, y_pred*np.std(y_train) + np.mean(y_train), label="Model")
        a = ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel('x', fontsize=FS)
        plt.ylabel('y', fontsize=FS)

        plt.show()    
                          
   
    def plot_prediction(self):
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.partition_lineardata()

        p = self.optimize()
        xn, yn = self.normalization(x_train,y_train)
        #the optimal value I got was below, therefore, I put the value down
        xt, yt = self.normalization(x_test,y_test)
        y_pred = np.dot(xn, p[0]) + p[1]
        y_pred1 = y_pred*np.std(y_train) + np.mean(y_train)
        y_pred2 = np.dot(xt, p[0]) + p[1]
        y_pred3 = y_pred2*np.std(y_test) + np.mean(y_test)
        fig, ax = plt.subplots()
        ax.plot(y_pred1, y_train, 'o', label='Training Set')
        ax.plot(y_pred3, y_test, 'o', label='Testing Set')

        ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel('y predicted', fontsize=FS)
        plt.ylabel('y data', fontsize=FS)

        plt.show()    
        
        
    #Logistic regression
    def partition_logdata(self):
        x = self.age
        y = self.weight

        x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size = 0.8)
        x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size = 0.5)

        return x_train, x_valid, x_test, y_train, y_valid, y_test

    def loss_log(self, p):
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.partition_logdata()
        nx,ny = self.normalization(x_train,y_train)
        MSE = 0
        for i in range(len(nx)):
            MSE += ((p[0] / (1 + np.e**((p[2] - nx[i])/p[1]))) + p[3] - ny[i])**2

        return MSE    
    
    def optimize_log(self):
         #TRAIN MODEL USING SCIPY OPTIMIZER
         po=np.random.uniform(0.5,1.,size=4)
         res = minimize(self.loss_log, po, method='Nelder-Mead')
         return res.x
    
    def plot_log(self):
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.partition_logdata()
        fig, ax = plt.subplots()
        ax.plot(x_train, y_train, 'o', label='Training Set')
        ax.plot(x_valid, y_valid, 'o', label='Validation Set')
        ax.plot(x_test, y_test, 'o', label='Testing Set')
        p = self.optimize_log()
        xn, yn = self.normalization(x_train,y_train)
        #the optimal value I got was below, therefore, I put the value down
        y_pred = p[3] + p[0] / (1 + np.e**((p[2] - xn)/p[1]))
        ax.plot(x_train, y_pred*np.std(y_train) + np.mean(y_train), label="Model")
        a = ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel('x', fontsize=FS)
        plt.ylabel('y', fontsize=FS)

        plt.show()    


        
    #classfication
    def partition_logisticdata(self):
        x = self.weight
        y = self.is_adult

        x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size = 0.8)
        x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size = 0.5)

        return x_train, x_valid, x_test, y_train, y_valid, y_test

    def loss_logistic(self, p):
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.partition_logisticdata()
        nx,ny = self.normalization(x_train,y_train)
        y = p[3] + p[0] / (1 + np.e**((p[2] - nx)/p[1]))
        MSE = 0
        for i in range(len(nx)):
            MSE += ((p[0] / (1 + np.e**((p[2] - nx[i])/p[1]))) + p[3] - ny[i])**2

        return MSE    
    
    def optimize_logistic(self):
         #TRAIN MODEL USING SCIPY OPTIMIZER
         po=np.random.uniform(0.5,1.,size=4)
         res = minimize(self.loss_logistic, po, method='Nelder-Mead')
         return res.x
    
    def plot_logistic(self):
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.partition_logisticdata()
        fig, ax = plt.subplots()
        ax.plot(x_train, y_train, 'o', label='Training Set')
        ax.plot(x_valid, y_valid, 'o', label='Validation Set')
        ax.plot(x_test, y_test, 'o', label='Testing Set')
        p = self.optimize_logistic()
        xn, yn = self.normalization(x_train,y_train)
        #the optimal value I got was below, therefore, I put the value down
        y_pred = p[3] + p[0] / (1 + np.e**((p[2] - xn)/p[1]))
        ax.plot(x_train, y_pred*np.std(y_train) + np.mean(y_train), label="Model")
        a = ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel('x', fontsize=FS)
        plt.ylabel('y', fontsize=FS)

        plt.show()    
                          
   
                          
   
    
d = Data('weight.json')
print(d.optimize())
d.plot_linear()
d.plot_log()
d.plot_logistic()
d.plot_prediction()
plt.show()




