#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 00:02:19 2021

@author: yangbo
"""

import statistics
from statistics import mean
import pandas as pd
from sklearn.model_selection import train_test_split  
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt

class Data:
    def __init__(self, filename):
        self.data = pd.read_json(filename)
    
    def show_data(self):
        return self.data
    
    def normalize_data(self, df):
        u_x = mean(df['x'])
        u_y = mean(df['y'])
        s_x = statistics.pstdev(df['x'])
        s_y = statistics.pstdev(df['y'])
        df['x'] = [(i - u_x)/s_x for i in df['x']]
        df['y'] = [(i - u_y)/s_y for i in df['y']]
        return df


    def partition_lineardata(self):
        #partition for linear data, only keep age less than 18 data points
        df = self.data
        age_18 = df['x'] <= 18
        x = df['x'][age_18]
        y = df['y'][age_18]

        x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size = 0.8)
        x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size = 0.5)

        return x_train, x_valid, x_test, y_train, y_valid, y_test

    
    def partition_data(self):
        #partition for logistic regression with x is age and y is weight
        df = self.data

        x_train, x_rem, y_train, y_rem = train_test_split(df['x'], df['y'], train_size = 0.8)
        x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size = 0.5)
        return x_train, x_valid, x_test, y_train, y_valid, y_test


    def partition_logdata(self):
        #partition for logistic regression with x is weight and y is is_adult
        df = self.data

        x_train, x_rem, y_train, y_rem = train_test_split(df['y'], df['is_adult'], train_size = 0.8)
        x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size = 0.5)
        return x_train, x_valid, x_test, y_train, y_valid, y_test


    def model_linear(self, x,p):
        #define linear model equation
        y = x*p[0] + p[1]
        return y
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def model_logistic(self, x,p):
        #define logistic regression equation
        y = p[0]+p[1]*(1.0/(1.0 + np.exp(-(x-p[2])/(p[3]+0.00001))))
        return y     

    def loss_linear(self, p):
        #calculate mse for linear regression 
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.partition_lineardata()
        y_pred = self.model_linear(x_train, p)
        total = y_train - y_pred
        MSE = np.sqrt(np.sum(np.power(total, 2)))/y_train.shape[0]
        loss = MSE
        return loss

    def loss_log(self, p):
        #calculate mse for logistic regression 
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.partition_data()
        #y_pred = self.model_logistic(x_train, p)
        #total = y_train - y_pred
        #MSE = np.sqrt(np.sum(np.power(total, 2)))/y_train.shape[0]
        #loss = MSE
        left = np.multiply(-y_train, np.log(self.model_logistic(x_train, p)))
        right = np.multiply(1-y_train, np.log(1-self.model_logistic(x_train, p)))
        loss = np.sum(left - right) / len(x_train)
        return loss

    def loss_logistic(self, p):
        #calculate mse for logistic regression 
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.partition_logdata()
        y_pred = self.model_logistic(x_train, p)
        total = y_train - y_pred
        MSE = np.sqrt(np.sum(np.power(total, 2)))/y_train.shape[0]
        loss = MSE
        return loss    
    
    def optimize(self,NFIT):
         #TRAIN MODEL USING SCIPY OPTIMIZER
        po=np.random.uniform(0.5,1.,size=NFIT)
        res = fmin(self.loss_linear, po)    
        return res
    
    def optimize_log(self, NFIT):
        po=np.random.uniform(0.5,1.,size=NFIT)
        res = fmin(self.loss_log, po)    
        return res        
    
    def optimize_logistic(self, NFIT):
        po=np.random.uniform(0.5,1.,size=NFIT)
        res = fmin(self.loss_logistic, po)    
        return res        

        
    def plot_linear(self):
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.partition_lineardata()
        fig, ax = plt.subplots()
        ax.plot(x_train, y_train, 'o', label='Training Set')
        ax.plot(x_valid, y_valid, 'o', label='Validation Set')
        ax.plot(x_test, y_test, 'o', label='Testing Set')
        #the optimal value I got was below, therefore, I put the value down
        ax.plot(x_train, 7.8579*x_train + 10.4873, label="Model")
        ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel('x', fontsize=FS)
        plt.ylabel('y', fontsize=FS)

        plt.show()
        
    def plot_data(self):
        #plotting the result with x value age, y value weight in logistic regression
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.partition_data()
        fig, ax = plt.subplots()
        ax.plot(x_train, y_train, 'o', label='Training Set')
        ax.plot(x_valid, y_valid, 'o', label='Validation Set')
        ax.plot(x_test, y_test, 'o', label='Testing Set')
        ax.plot(x_train, self.model_logistic(x_train, self.optimize_logistic(4)), label="Model")
        ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel('x', fontsize=FS)
        plt.ylabel('y', fontsize=FS)

        plt.show()
        
    def plot_log(self):
        #plotting the result with x value weight and y value is_adult
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.partition_logdata()
        fig, ax = plt.subplots()
        ax.plot(x_train, y_train, 'o', label='Training Set')
        ax.plot(x_valid, y_valid, 'o', label='Validation Set')
        ax.plot(x_test, y_test, 'o', label='Testing Set')
        ax.plot(x_train, self.model_logistic(x_train, self.optimize_logistic(4)), label="Model")
        ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel('x', fontsize=FS)
        plt.ylabel('y', fontsize=FS)

        plt.show()
                       
   
    def plot_prediction(self):
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.partition_lineardata()
        y_pred = x_train*7.8579 + 10.4873
        y_pred1 = x_valid*7.8579 + 10.4873
        y_pred2 = x_test*7.8579 + 10.4873
#        y_pred = self.model_linear(x_train, self.optimize(2))
 #       y_pred1 = self.model_linear(x_valid, self.optimize(2))
  #      y_pred2 = self.model_linear(x_test, self.optimize(2))

        fig, ax = plt.subplots()
        ax.plot(y_pred, y_train, 'o', label='Training Set')
        ax.plot(y_pred1, y_valid, 'o', label='Validation Set')
        ax.plot(y_pred2, y_test, 'o', label='Testing Set')

        ax.legend()
        FS=18   #FONT SIZE
        plt.xlabel('y predicted', fontsize=FS)
        plt.ylabel('y data', fontsize=FS)

        plt.show()
        
        
d = Data('weight.json')
print(d.plot_linear())
print(d.plot_data())
print(d.plot_log())
print(d.plot_prediction())


