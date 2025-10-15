#Code by Doga Dokuz, 2025
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# create the feed-forward neural network for XOR function
class XOR(nn.Module):
    def __init__(self,input,hidden,output):
        super(XOR, self).__init__()
        self.hidden = nn.Linear(input, hidden)  # input layer to hidden layer
        self.output = nn.Linear(hidden, output)  # hidden layer to output layer
        self.relu = nn.ReLU() #activation function
    # int = 2, hidden1= 2 ,output =1
    def forward(self, x):
        x_in = self.relu(self.hidden(x)) # input x goes to hidden first then activation function is applied on it
        x_out = self.output(x_in)  # after the activation function, we just out put it
        return x_out

# we need to define the training data for XOR function
X = torch.zeros((4, 1, 2))
y = torch.zeros((4, 1, 1))

# our XOR inputs
X[0, 0] = torch.tensor([0.0, 0.0])
X[1, 0] = torch.tensor([0.0, 1.0])
X[2, 0] = torch.tensor([1.0, 0.0])
X[3, 0] = torch.tensor([1.0, 1.0])

# our XOR outputs with corresponding X inputs
y[0, 0] = torch.tensor([0.0])
y[1, 0] = torch.tensor([1.0])
y[2, 0] = torch.tensor([1.0])
y[3, 0] = torch.tensor([0.0])

# creating a training model for our XOR function
def train_model():
    model = XOR(2,2,1) #input,hidden,output
    criterion = nn.MSELoss() #we are using the mean squared error function as our loss function 
    optimizer = optim.SGD(model.parameters(), lr=0.01) #later we are calculating the stochastic gradient descent of our loss function to find the optimal parameters
    losses = [] #defining a list to store the losses to plot them later
    model.train()
    for epoch in range(1500):
        optimizer.zero_grad()
        predicted_output = model(X.view(4, 2))  # Reshape input
        
        loss = criterion(predicted_output.view(4, 1, 1), y)  # Ensure shapes match
        loss.backward()
        optimizer.step() 
        losses.append(loss.item())
        #optimizer.zero_grad()
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    return model, losses

# in the analytical example of the lecture, we discussed to add a nonlinear function by a non-linear hidden layer where we can still have linearly dependent output layer mapped it into different space
# The difference between 100 and 10000 epoch is that in 100, it is not possible to see a total convergence, it can be seen that the MSE is going down but we can not say anything if it is starting to converge or not. However, in the epoch case of 1000, we can see some convergence
# If it is around 10 000 epochs, we see like a ReLU type of function where the graph gets from the top to bottom with fast dip and then a constant line. With many additional networks, we all see that they start to dip then converge very fast, similar to ReLU function
# How do the parameters found by the network training algorithm compare to what we discussed in the lecture?
# We found the parameters with the function of optim.SGD(model.parameters) function where we decided on our learning rate as 0.01, if the learning rate is too small, it can be stuck to flat regions of the ReLU function and makes it harder see the convergence/optimal point

# now we train 10 models with different initializations, we observe that after 1500 epochs all the other neural networks converge 

model,losses = train_model()
plt.plot(losses,label="first_training_data")
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Training Loss for XOR Neural Network (10 Runs)')
plt.legend()
plt.show()

for i in range(9):
    model, losses = train_model() #storing the model and the losses calculated along training our model
    #all_losses.append(losses)
    print(f"Model {i+1} parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
            print('------------------------')
    plt.plot(losses,label=f"Run{i+2}")
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss for XOR Neural Network (10 Runs)')
    plt.legend()
plt.show()
