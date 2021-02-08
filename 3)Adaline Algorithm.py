# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:50:55 2020

@author: mehme
"""

import matplotlib.pyplot as plt 
import numpy as np # numpy is a fundamental library/package for scientific computing in Python 

class Adaline_single_neuron():
    
    def __init__(self, input_size, lr=0.01, epochs=3):
        # when we use self.anything , we can access the anything from inside the same class
        self.W = np.zeros(input_size+1) # add one size for bias(w0)
        self.epochs = epochs
        self.lr = lr # lr=learning_rate , we say eta in our lessons
    
   
 
    def fit(self, X, d):
        self.weights_visualization = []
        for _ in range(self.epochs): # epochs determines number of iteration
            for i in range(d.shape[0]): # d.shape ==> (40,1), this means 40 row,1 column , d.shape[0] == number of total row
                x = np.insert(X[i], 0, 1) # make 1 (bias*x0 and x0 always 1) the first value and then takes x1 and x2  
                V_pred = self.predict(x)
                e = d[i] - V_pred # error = desired output - real output . error artık desired_V - bulduğumuz V
                self.W = self.W + self.lr * e * x    # w(n) + learning_rate*[desired-our_result] * (x1,x2,..xn)
                
                self.weights_visualization.append(self.W)
                
        
        fig,ax = plt.subplots()           
        ax.set(ylabel= "Weights", xlabel="Iter")
        plt.plot(self.weights_visualization)
        plt.legend(["W0","W1","W2"], loc="right")
        plt.show()

        
        
    def predict(self, x):
        expected_V = self.W.T.dot(x) # W.T ==> transpose of W. dimension of transpose of W = 1x3, dimension of X = 3x1  this process will show the w0*x0+w1*x1...
        #expected_V = np.dot(x, self.W.T) aynısı
        return expected_V
 
    def transfer_func(self, x): # activation/transfer func. that we use 
            z = 1/(1 + np.exp(-x)) # sigmoid function
            return z
        
                
    def output(self,X):
        y_values = []
        for i in range(len(X)): 
            x = np.insert(X[i], 0, 1)  
            output = self.W.T.dot(x)           
            y_values.append(self.transfer_func(output))
        return y_values 
            
      
    
if __name__ == '__main__':                
    
    np.random.seed(0) # random numbers don't change in every run
    first_group = np.random.uniform(low=5, high=10, size=(20,2)) # first group data 
    sampl2 = np.random.uniform(low=-5, high=0, size=(20,1)) # x1 values of second group data 
    sampl3 = np.random.uniform(low=0, high=3, size=(20,1)) # x2 values of second group data 
    
    second_group = np.concatenate((sampl2, sampl3), axis=1) # merge x2 values with x1 values of second group data
    
    first_group_desired = []
    second_group_desired = [] 
    for i in range(len(first_group)): # v_desired   = -50 + x1*10 + x2*5
        first_group_desired.append(-50 + (first_group[i][0])*10 + (first_group[i][1])*5)
        second_group_desired.append(-50 + (second_group[i][0])*10 + (second_group[i][1])*5)
        i += 1
    
    first_group_desired = np.asarray(first_group_desired)
    first_group_desired = first_group_desired.reshape(20,1)
    second_group_desired = np.asarray(second_group_desired)
    second_group_desired = second_group_desired.reshape(20,1)
    
    first_group = np.concatenate((first_group, first_group_desired), axis=1) # merge first group datas and desire outputs
    second_group = np.concatenate((second_group, second_group_desired), axis=1) # merge second group datas and desire outputs
    
    
    concenate = np.concatenate((first_group, second_group), axis=0) # merge first group datas and second group datas , vertically
    np.random.shuffle(concenate) # mixed datas' places
    
    
    desired_ = concenate[:,2] # desired_values
    concenate = np.delete(concenate, 2, axis=1) # deleted the desired values , just contain first group datas and second group datas
    
    perceptron = Adaline_single_neuron(input_size=2) 
    
    perceptron.fit(concenate, desired_)
    
    print(perceptron.W)   
    print("Bias:",perceptron.W[0])
    print("W1 and W2: ", perceptron.W[1:3] )
    
    
    
    
    fig,ax = plt.subplots()
    ax.set(title = "Dataset", ylabel= "x2 values",xlabel="x1 values")
    plt.scatter(concenate[:,0],concenate[:,1])
    plt.show()
    
    print("Y values: ",perceptron.output(concenate))  # print what the output of ith row is         
    
    fig,ax = plt.subplots()  # visualize what the output of ith row is        
    ax.set(xlabel="ith values", ylabel= "Outputs")
    plt.scatter(range(40),perceptron.output(concenate))
    plt.show()