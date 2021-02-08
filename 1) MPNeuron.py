# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:35:48 2020

@author: mehme
"""

import sklearn.datasets
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

breast_cancer = sklearn.datasets.load_breast_cancer()

X = breast_cancer.data
Y = breast_cancer.target

#Converting the data to Pandas DataFrame
data = pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
data["Class"] = breast_cancer.target
print(data.head())

from sklearn.model_selection import train_test_split

X = data.drop('Class', axis=1)
Y = data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify = Y)

#converting the input features to a binary format, all inputs must be binary in MP Neuron Model
#So we need to convert the continuous features into binary format. To achieve this, we will use pandas.cut function to split all the features into 0 or 1 in one single shot. 
X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[1,0])
X_binarised_test = X_test.apply(pd.cut, bins=2, labels=[1,0])


#To create a MP Neuron Model we will create a class and inside this class, we will have three different functions:
#model function — to calculate the summation of the Binarized inputs.
#predict function — to predict the outcome for every observation in the data.
#fit function — the fit function iterates over all the possible values of the threshold b and find the best value of b, such that the loss will be minimum.

class MPNeuron:
    
    
    def __init__(self):
        self.b = None
        
    def model(self,x):
        return (sum(x) >= self.b)
    
    def predicts(self,X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
            
        return np.array(Y)
    
    def fit(self,X,Y):
        accuracy = {}
        
        for b in range(X.shape[1]+1):            
            self.b = b
            Y_pred = self.predicts(X)
            accuracy[b] = accuracy_score(Y_pred,Y)
        
        best_b = max(accuracy,key=accuracy.get)
        self.b = best_b
        
        print('Optimal value of b is', best_b)
        print('Highest accuracy is', accuracy[best_b])
        
#Calling the class MPNeuron

mp_neuron = MPNeuron()

#Calling the fit method inside the class on the training data
mp_neuron.fit(X_binarised_train, Y_train)


#testing the model on the test data.
Y_test_pred = mp_neuron.predict(X_binarised_test)
accuracy_test = accuracy_score(Y_test_pred, Y_test)

#print the accuracy of the test data
print(accuracy_test)
#%%
# MP-NEURON FOR "AND GATE" "OR GATE"
a = [0,1,0,1]
b = [0,0,1,1]
z= [0,1,1,1] # Target Vector

# weights are 1 in a MP Neuron
weights = 1

print("Determine a threshold: ")
threshold = float(input())

bias = 0
while True:
    y_pred = []
    for i in range(len(a)):
        g_func_sum =  bias+a[i]*weights+b[i]*weights      
        i += 1
        
        if (g_func_sum >= threshold):
            y_pred.append(1)
            print("First values' output is {}",1)
        else:
            y_pred.append(0)
            print("First values' output is {}",0)
    
    if y_pred == z:
        print("Procesess is true. Target values: {} predicted Values: {}".format(z,y_pred))
        break
    else:
        print("Please change the bias: ")
        bias = float(input())

      