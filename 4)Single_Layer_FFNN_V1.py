from random import random
import numpy as np
import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore', 'overflow')

class slffnn():
    
     def __init__(self, input_size, output_neuron,  lr=0.01, epochs=10): 
        
       
        self.output_layer_weights = [{'weights_output_neuron{}'.format(i+1):[random() for i in range(input_size + 1)]} for i in range(output_neuron)]
        # we created output layers' neurons' weights. There are weights as much (input size + 1(bias)) as for each neuron 
        
        self.epochs = epochs
        self.lr = lr 
        
    
     def fit(self, X, desired_Vs):
        
        
        for _ in range(self.epochs): 
            
            for i in range(X.shape[0]): # loop for each data row
                
                x = np.insert(X[i], 0, 1) 

                ys = []
                errors = []
                calc_local_gradients = []
                weight_values = []
                
                for i in range(len(self.output_layer_weights)):
                    
                    ys.append(self.predict(x,self.output_layer_weights[i]) ) # holding "y" values in ys 
                
                
                for i in range(len(desired_Vs)):     # calculate errors for each output
                   errors.append(desired_Vs[i] - ys[i])

                    
                for i in range(len(self.output_layer_weights)): # # calculate local gradients for each output
                    calc_local_gradients.append(self.local_gradients(errors[i],ys[i])) 
                    
   
                    for key, value in dict.items(self.output_layer_weights[i]):
                        """ weights are holding in a data structure that called dictionary,
                        we seperate weight values and neuron names then we extract weight values due to this function """
                        weight_values.append(value)
                         
                    w_array_t = np.array(weight_values) # extracts weights in "time t" from dictionary, and writes to this variable
                    
                    self.output_layer_weights[i] = {'weights_of_neuron{}'.format(i+1):w_array_t[i] + self.lr*calc_local_gradients[i] *x} 
                    
            
            
            return print(self.output_layer_weights)
            
            
        
     def predict(self, x,w):
         
         ws = []
         for key, value in dict.items(w):
             ws.append(value)
             
         w_array = np.array(ws)
         expected_V1 = w_array.dot(x)
         y1 = self.transfer_func(expected_V1)
         return y1
 
     def transfer_func(self, x): 
             z = 1/(1 + np.exp(-x)) 
             return z
        
     def local_gradients(self,errors,ys):
        local_gradient = errors*(ys*(1-ys))
        return local_gradient
            



if __name__ == '__main__':                
    
    
    amount_of_data = int(input("How many data row do you want to create ?")) 
    
    inputSize = int(input("Please enter the input size without bias : "))
 
    np.random.seed(0) # random numbers don't change in every run
    data = np.random.uniform(low=-10, high=10, size=(amount_of_data,inputSize))
    
    
    
    output_neuron_size = int(input("Please enter the output neuron size "))
    
    desired = []
    for i in range(output_neuron_size):
        desired_value = int(input("Please enter desired output value of output neuron{} ".format(i+1)))
        desired.append(desired_value)
        
    desired = np.array(desired)
    
    
    
    
    slf = slffnn(inputSize,len(desired))  # kaç tane desired V varsa o kadar output neuron olması lazım, bu yüzden buraya len(desired_) yazdık
    
    slf.fit(data, desired)
    
    

    