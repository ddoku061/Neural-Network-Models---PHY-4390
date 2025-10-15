#Code by Doga Dokuz
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt 

with open('q1_data_20.npy', 'rb') as f:
    x_20 = np.load(f)
    y_20 = np.load(f)
with open('q1_data_30.npy', 'rb') as f:
    x_30 = np.load(f)
    y_30 = np.load(f)
with open('q1_data_40.npy', 'rb') as f:
    x_40 = np.load(f)
    y_40 = np.load(f)
with open('q1_data_80.npy', 'rb') as f:
    x_80 = np.load(f)
    y_80 = np.load(f)
with open('q1_data_160.npy', 'rb') as f:
    x_160 = np.load(f)
    y_160 = np.load(f)    
with open('q1_data_200.npy', 'rb') as f:
    x_200 = np.load(f)
    y_200 = np.load(f) 

def polynominal_regression_model(x,y,k=2):
    #n = len(x)
    m = len(x)//2
    n = 20 #length of one vector in x, in the w matrix, only k changes
    x_train = x[:m]
    y_train = y[:m]
    x_test = x[m:]
    y_test = y[m:]
    print(f'{len(x)} Data Points --------------------- k={k} ----------------')
    w_initial = np.random.uniform(-0.1, 0.1, size=(n, k)) #n is fixed to 20 since that is the length of X1 = [x1,...,x20]
    
    def polynominal(w,x_train,k):
        y_hat = np.zeros(x_train.shape[0])
        for i in range(1,k+1):
            y_hat += np.transpose(x_train**(i) @ (w[:,i-1])) #we are getting the columns for W matrix, because if k = 1, we need w (1) = [w(1)_1 -> w(1)_n]
        return y_hat
    
    def mse(y_train,y_train_prediction,w,m): #calculating the mean squared similar to the previous question
        MSE_Y = []
        for i in range(0,m):
            subst_y = ((y_train_prediction[i] - y_train[i]) ** 2) #find the error between predicted and expected y value
            MSE_Y.append(subst_y)
        mse_test = np.mean(MSE_Y)
        return mse_test
    
   
    def gradient(x, y, w, k): #as the gradient is derived in the attached image, we use the generalized form for gradient
        grad = np.zeros_like(w) #creating an array of 0's similar to the size of weight array
        y_pred = polynominal(w, x, k) #predicting y via the polynominal function
        
        for i in range(1, k + 1):
            grad[:, i - 1] = np.mean(2 * (y_pred - y)[:, np.newaxis] * (x ** i), axis=0) #taking the mean in the function of SGD
        return grad
    
    def stochastic_gradient_descent(x_train, y_train, w_initial,k, lr = 0.001, epochs = 1000, batch_size = 5):
        w_updated = w_initial # we are starting with the initial w, which will be updated in every epoch
        epoch_plot = [] #just to store epoch number to plot later in the code
        mse_train_plot = [] #just to store training mse value to plot later in the code
        mse_test_plot = [] #just to store testing mse value to plot later in the code
       
        print('Targeted y: \n {}'.format(y_test)) #initially printing what the expected value of y
        for epoch in range(epochs): #going through each epoch            
            indices = np.random.permutation(m) #doing permutation of numbers until m
            
            for i in range(0, m, batch_size): #pulling out mini-batch within the number of data points
                batch_indices = indices[i:i + batch_size] #redefining the indice array between 0,5
                x_batch = x_train[batch_indices] #assigning the indices to x and y to create new size of x and y array
                y_batch = y_train[batch_indices]
                grad = gradient(x_batch, y_batch, w_updated,k) #calculating gradient to update weights relating to the new array of batch 
            w_updated -= (lr * grad) #updating the weight based on the output of the gradient
            
            y_train_pred = polynominal(w_updated,x_train,k) #predicting the training y-value with the updated weights and our training x data
            mse_train_value = mse(y_train,y_train_pred,w_updated,m) #calculating the error with in the training data, for fun
            mse_train_plot.append(mse_train_value) #storing them to plot them later
            
            y_test_pred = polynominal(w_updated, x_test, k) #predicting the testing y-value with the updated weights and our test x data
            mse_test_value = mse(y_test,y_test_pred,w_updated,m) #calculating the error with in the test data, predicted by the updated weight in each epoch
            mse_test_plot.append(mse_test_value) #storing them to plot them later
            epoch_plot.append(epoch) #storing them to plot them later
            
            if epoch % 50 == 0 or epoch == epochs - 1: #every 50 epochs, it gives the our epoch progress, mse progress and the predicted y test value
                print(f"Epoch {epoch + 1}/{epochs}, Test Error: {mse_test_value:.4f}------")
                print(f'Predicted tested y: \n {y_test_pred} at epoch {epoch + 1}/{epochs}') 
        return epoch_plot,mse_test_plot, mse_train_plot
    

    #calling the SGD function to perform on the existing data sets    
    stochastic_plot = stochastic_gradient_descent(x_train, y_train,w_initial, k, lr = 0.001, epochs = 1000, batch_size = 5)
    plt.title(f"{m} Training Data Points (Total:{m*2}), \n Mean Squared Error vs Epoch in Log Scale")
    plt.plot(stochastic_plot[0], stochastic_plot[1], label = 'Test Data, k:'+str(k))  
    plt.plot(stochastic_plot[0], stochastic_plot[2], label = 'Train Data, k:'+str(k)) 
    plt.xlabel("# of epochs")
    plt.ylabel("Mean Squared Error")
    plt.yscale("log")
    plt.legend()
    #plt.show()
    
#easy way to call the general function to implement SGD process and plot them
data_storage = ([x_20,y_20],[x_30,y_30],[x_40,y_40],[x_80,y_80],[x_160,y_160],[x_200,y_200])
for data in data_storage:
    plt.figure(figsize=(8, 6))
    plt.grid()
    for j in range(2,4):
        polynominal_regression_model(data[0],data[1],j)
    plt.show()    
