# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:51:17 2020

@author: mehme
"""

import matplotlib.pyplot as plt 
import numpy as np # numpy is a fundamental library/package for scientific computing in Python 

class Perceptron_single_neuron():
    
    def __init__(self, input_size, lr=0.01, epochs=3):
        # when we use self.anything , we can access the anything from inside the same class
        self.W = np.zeros(input_size+1) # add one size for bias(w0)
        self.epochs = epochs
        self.lr = lr # lr=learning_rate , we say eta in our lessons
    
   
 
    def fit(self, X, d):
        self.weightss = []
        for _ in range(self.epochs): # epochs determines number of iteration
            for i in range(d.shape[0]): # d.shape ==> (40,1), this means 40 row,1 column , d.shape[0] == number of total row
                x = np.insert(X[i], 0, 1) # make 1 (bias*x0 and x0 always 1) the first value and then takes x1 and x2  
                y_pred = self.predict(x)
                e = d[i] - y_pred # error = desired output - real output
                self.W = self.W + self.lr * e * x    # w(n) + sabitdeğer*[desired-our_result] * (x1,x2,..xn)
                
                self.weightss.append(self.W)
                
                #print(self.weightss)
                #fig,ax = plt.subplots()           
                #ax.set(ylabel= "Weights", xlabel="Iter")
                #plt.plot(self.weightss)
                #plt.show()
 
        
        #print(self.weightss)
        fig,ax = plt.subplots()           
        ax.set(ylabel= "Weights", xlabel="Iter")
        plt.plot(self.weightss)
        plt.legend(["W0","W1","W2"], loc="upper rights")
        plt.show()
        
        
        
        
    def predict(self, x):
        s = self.W.T.dot(x) # W.T ==> transpose of W. dimension of transpose of W = 1x3, dimension of X = 3x1  this process will show the w0*x0+w1*x1...
        #s = np.dot(x, self.W.T) aynısı
        pred = self.transfer_func(s) # takes predicted value from transfer function
        return pred
 
    def transfer_func(self, x): # activation/transfer func. that we use 
          if x> 0:   # signum function 
              return 1
          else:
              return -1
    
if __name__ == '__main__':                
    
    np.random.seed(0) # random numbers don't change in every run
    first_group = np.random.uniform(low=5, high=10, size=(20,2)) # first group data 
    sampl2 = np.random.uniform(low=-5, high=0, size=(20,1)) # x1 values of second group data 
    sampl3 = np.random.uniform(low=0, high=3, size=(20,1)) # x2 values of second group data 
    
    second_group = np.concatenate((sampl2, sampl3), axis=1) # merge x2 values with x1 values of second group data
    
    first_group_desired = np.ones((20,1)) # first group values' desired outputs
    second_group_desired = np.ones((20,1))*(-1) # second group values' desired outputs
    first_group = np.concatenate((first_group, first_group_desired), axis=1) # merge first group datas and desire outputs
    second_group = np.concatenate((second_group, second_group_desired), axis=1) # merge second group datas and desire outputs
    
    
    concenate = np.concatenate((first_group, second_group), axis=0) # merge first group datas and second group datas , vertically
    np.random.shuffle(concenate) # mixed datas' places
    
    
    desired_ = concenate[:,2] # desired_values
    concenate = np.delete(concenate, 2, axis=1) # deleted the desired values , just contain first group datas and second group datas
    
    perceptron = Perceptron_single_neuron(input_size=2) 
    
    perceptron.fit(concenate, desired_)
    
    print(perceptron.W)   
    print("Bias:",perceptron.W[0] )
    print("W1 and W2: ", perceptron.W[1:3] )
    
    
    
    
    fig,ax = plt.subplots()
    ax.set(title = "Dataset", ylabel= "x2 values",xlabel="x1 values")
    plt.scatter(concenate[:,0],concenate[:,1])
    plt.show()
    
    
