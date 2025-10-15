#Code by Doga Dokuz, 2025

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Load datasets
files = ["corr_nn.txt", "corr_nnn.txt", "mz.txt", "mx.txt"]
data = [np.loadtxt(f) for f in files]
time_steps = np.arange(0, 2.01, 0.01)  # Time from 0 to 2 with step 0.01
time_steps_auto = np.arange(0, 201, 1)
# Plot time evolution
plt.figure(figsize=(10, 6))
labels = ["Nearest Neighbor Corr", "Next-Nearest Neighbor Corr", "Magnetization (Z)", "Magnetization (X)"]
for i, d in enumerate(data):
    plt.plot(time_steps, d, label=labels[i])
plt.xlabel("Time (t)")
plt.ylabel("Expectation Values")
plt.legend()
plt.grid()
plt.title("Time Evolution of Observables")
plt.show()

#converting our data to tensors
data = torch.tensor(np.array(data).T, dtype=torch.float32)  # Shape (200, 4)

#creating a vanilla recurrent neural network
class VanillaRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=4): #input size is 1, hidden size is 32 and output size is 4
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True) #we are using the RNN library from pytorch to create the rnn part of the vanilla 
        self.fc = nn.Linear(hidden_size, output_size)  #linearly connect the hidden and output layers

    def forward(self, x, h=None):
        out, h = self.rnn(x, h)  #the RNN library takes the h at time 0 and the input sequence, then returns the h at time t and the output sequence
        out = self.fc(out) #using very last time (t) step output
        return out, h

# GRU RNN
class GRUNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=4):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        out = self.fc(out)
        return out, h

#defining a trainging function to train neural network models (separate for vanilla RNN and GRU)
def train(model, data, epochs=2000, lr=0.001,label='model'):
    optimizer = optim.Adam(model.parameters(), lr=lr) #selecting our optimizer as ADAM 
    criterion = nn.MSELoss() #selecting our loss function as mean squared error (MSE)
    model.train() #setting the model to the training mode
    input_seq = data[:50, 0].view(1, -1, 1)  #nearest-neighbor correlation as input wwith first 50 time steps
    
    target_seq = data[1:51]  #time steps for all four observables as target outputs
    
    hidden = torch.zeros(1, 1, model.hidden_size) #initializing the hidden state with zeros 
    
    for epoch in range(epochs):
        optimizer.zero_grad() #setting the gradients to zero
        output, _ = model(input_seq, hidden) #taking the sequence output of our model 
        loss = criterion(output, target_seq) #computing the loss function of mse
        loss.backward() #backpropagating
        optimizer.step() #updating the weights
        #printing the loss function every 100 epochs
        if epoch % 100 == 0:
            print(f"{label} Model, Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return model

#training the vanilla RNN and GRU models
rnn_model = VanillaRNN()
rnn_model_t = train(rnn_model, data, label='Vanilla RNN')

gru_model = GRUNet()
gru_model_t = train(gru_model, data, label= 'GRU RNN')

#predicting the full sequence with trained models
def predict(model, data):
    model.eval()
    hidden = torch.zeros(1, 1, model.hidden_size)  #initializing the hidden state with zeros 
    input_seq = data[:200, 0].view(200, 1, 1)  #full nearest-neighbor correlation input (the batch is picked 200 because of the question)
    predictions = [] #store our predictions for each time step
    
    for t in range(200):  #predict step by step
        output, hidden = model(input_seq[t:t+1], hidden) #predicting the output and hidden state
        predictions.append(output.detach().numpy().flatten()) 
        #.detach() creates a tensor that shares storage with tensor that does not require grad. It detaches the output from the computational graph. So no gradient will be backpropagated along this variable.
    return np.vstack(predictions)  #stacking the predictions into shape (200, 4)


def autoregressive_predict(model, data, start_steps=50, total_steps=200): #creating a autoregressive prediction model
    model.eval()
    hidden = torch.zeros(1, 1, model.hidden_size)  #initializing the hidden state
    #input_seq = data[:start_steps, 0].view(start_steps, 1, 1)  #we want to start with 50 known inputs
    predictions = list(data[:start_steps].numpy())  #storing the known inputs first to our predictions

    #iterating over the time steps
    for t in range(start_steps, total_steps):
        input_tensor = torch.tensor([[predictions[-1][0]]], dtype=torch.float32).view(1, 1, 1)  #taking the last known input
        output, hidden = model(input_tensor, hidden)  #predicting each step
        predictions.append(output.detach().numpy().flatten())  #storing the predicted output 
    return np.array(predictions) #returning the predictions as an array

#I was not sure if we were asked to model it from time steps of 50 or 0 but here is the function to model it from time step 0 below: 

# def autoregressive_predict(model, data, total_steps=200): #creating a autoregressive prediction model
#     hidden = torch.zeros(1, 1, model.hidden_size)
#     input_val = torch.tensor([[data[0, 0]]], dtype=torch.float32).view(1, 1, 1)  # Start with NN correlation at t=0
#     predictions = []

#     for t in range(total_steps):
#         output, hidden = model(input_val, hidden)
#         output_np = output.detach().numpy().flatten()
#         predictions.append(output_np)
#         input_val = torch.tensor([[output_np[0]]], dtype=torch.float32).view(1, 1, 1)  # Use predicted NN as next input

#     return np.array(predictions)


rnn_pred = predict(rnn_model_t, data) #getting our predictions for vanilla RNN
gru_pred = predict(gru_model_t, data) #getting our predictions for GRU
#adjusting time_steps to match the predicted values
time_steps_pred = time_steps[1:]  #now our tensor has 200 values

plt.figure(figsize=(12, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(time_steps, data[:, i], label="Exact")  # Exact data has 201 points
    plt.plot(time_steps_pred, rnn_pred[:, i], '--', label="Vanilla RNN")  # Predicted has 200 points
    plt.plot(time_steps_pred, gru_pred[:, i], ':', label="GRU")  # Predicted has 200 points
    plt.xlabel("Time (t)")
    plt.ylabel('Predicted Time Evolution')
    plt.title(labels[i])
    plt.legend()
plt.suptitle("Comparison of Exact and Predicted Time Evolution")
plt.show()

rnn_auto = autoregressive_predict(rnn_model, data) #getting our autoregressive predictions for vanilla RNN
gru_auto = autoregressive_predict(gru_model, data) #getting our autoregressive predictions for vanilla RNN
time_steps_p_auto = time_steps_auto[1:]  
plt.figure(figsize=(12, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(time_steps_auto, data[:, i], label="Exact")  # Exact data has 201 points
    plt.plot(time_steps_p_auto, rnn_auto[:, i], '--', label="Vanilla RNN")  # Predicted has 200 points
    plt.plot(time_steps_p_auto, gru_auto[:, i], ':', label="GRU")  # Predicted has 200 points
    plt.xlabel("Time (t)")
    plt.ylabel('Predicted Time Evolution')
    plt.title(labels[i])
    plt.legend()
plt.suptitle("Comparison of Exact and Autoregressively Predicted Time Evolution")
plt.show()
