
#This part of the code is provided by Stefanie Czischek to import the training data and bring it into the desired shape.
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from torchvision.transforms import ToTensor

# Download training data from open datasets
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
### Collect training data ###
# Turn training data and labels into numpy arrays as we won't use PyTorch in this part
X_data = training_data.data.numpy()
y_data = training_data.targets.numpy()

###########################
#Code by Doga Dokuz, 2025
###########################
# We are only interested in training examples that display a 5
training_batch = X_data[y_data == 5]

# We only take every second entry to reduce the dimension of the input data to 14x14 pixels
# And we randomly select 100 training examples
index = np.random.randint(0, np.shape(training_batch)[0], (100))
training_batch = training_batch[index,::2,::2]

# We flatten each input example into a one-dimensional array
training_batch = training_batch.reshape(np.shape(training_batch)[0], -1)

# Finally, we turn the input data into binary values by setting all values <0.5 to zero and all values >=0.5 to 1
# We achieve this by checking for every element whether the statement >= 0.5 is true (value 1) or false (value 0).
training_batch = (training_batch >= 0.5).astype(np.int_)



sig = nn.Sigmoid() #could not figure out np.array and tensor relation
#instead I implemented a function for sigmoid math function, where we want to use it as our activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#calculating probabilities for hidden neurons given visible neurons using a sigmoid function.
def sample_hidden(visible,weights,hidden_bias): #creating hidden samples and updating the new hidden configurations from the probability distribution
    prob_input = (weights.transpose() @ visible)+hidden_bias #calculating the energy function for probability distribution
    probability_hidden = sigmoid(prob_input) #calculating the probability with sigmoid fn
    return (np.random.rand(*probability_hidden.shape) <= probability_hidden).astype(int)

#calculating probabilities for visible neurons given hidden neurons using a sigmoid function.
def sample_visible(hidden,weights,visible_bias): #creating visible samples and updating the new visible configurations from the probability distribution
    
    prob_input = (weights @ hidden)+visible_bias #calculating the energy function for probability distribution
    probability_visible = sigmoid(prob_input) #calculating the probability with sigmoid fn
    return (np.random.rand(*probability_visible.shape) <= probability_visible).astype(int) #comparing

#creating a function for contrastive divergence to implement Gibbs Sampling
def contrastive_div(visible_neuron, weights, visible_bias, hidden_bias,lr=0.01):
    for visible in visible_neuron:
        
        visible = visible.reshape(-1,1) #we reshape the data for a easier matrix calculations
        
        v0 = visible #starting with random configuration of random visible neurons
        h0 = sample_hidden(v0,weights,hidden_bias) #drawing new hidden samples from the random configuration of visible
        v1 = sample_visible(h0,weights,visible_bias) #drawing new visible samples from the random configuration of hidden
        h1 = sample_hidden(v1,weights,hidden_bias) #repeating the draw of hidden samples to allow convergence in our model
        #the chain needs a several steps until it converges to sampling from the model distribution
        
        weights += lr * (np.outer(v0, h0) - np.outer(v1, h1)) #we are updating the weights with respect to v0,h0 and v1. h1
        visible_bias += lr * (v0 - v1) #we are updating the visible bias with samples that are drawn for v0 and v1
        hidden_bias += lr * (h0 - h1) #we are updating the hidden bias with samples that are drawn for h0 and h1
    
    return weights, visible_bias, hidden_bias 

class RestrictedBM: #creating our class for the restricted boltzmann machine
    def __init__(self,num_visible,num_hidden):
        self.visible = num_visible #196
        self.hidden = num_hidden #50
        self.weight = np.random.uniform(-0.05, 0.05, (num_visible, num_hidden)) #randomly generating the weights of the model
        self.visible_bias = np.random.uniform(-0.05, 0.05, (num_visible,1)) #randomly generating the visible bias of the model
        self.hidden_bias = np.random.uniform(-0.05, 0.05, (num_hidden,1)) #randomly generating the hidden bias of the model

def train_rbm(model,train_data,num_epochs): #defining a function to train our rbm
    for epoch in range(num_epochs): #training the model with a given number of epochs
        parameters = contrastive_div(train_data,model.weight,model.visible_bias,model.hidden_bias) #getting parameters from the contrastive divergence function that uses Gibbs Sampling
        model.weight = parameters[0] #modifying the model weights depending on the output of the contrastive divergence function
        model.visible_bias = parameters[1] #modifying the model visible biases depending on the output of the contrastive divergence function
        model.hidden_bias = parameters[2] #modifying the model hidden biases depending on the output of the contrastive divergence function
        if epoch % 10 == 0:
            print(f"Epoch {epoch} completed") #keeping track of the epoch number

def generate_image(model,test_data,num_visible,random): #creating a function to reconstruct an image
    
    new_image = np.empty((random,num_visible)) #creating an empty list
    for i in range(1000): #running over 1000 samples
        #x = samples[i]
        #x = x.reshape(-1,1)
        hidden = sample_hidden(test_data,model.weight,model.hidden_bias) #sampling hidden neurons from the model
        test_data = sample_visible(hidden,model.weight,model.visible_bias) #sampling visible neurons from the model
        new_shape_data = test_data.reshape(num_visible,) #reshaping the data to the desired shape
        new_image[i] = new_shape_data #appending the reshaped data to the empty list
        
    return new_image


num_visible = 196  # 14x14 pixels
num_hidden = 50
num_epochs = 50
random = 1000
# samples = np.random.rand(random, num_visible)
test_data = np.random.rand(num_visible,1)
model = RestrictedBM(num_visible,num_hidden)
train_rbm(model,training_batch,num_epochs)

image = generate_image(model,test_data,num_visible,random) #generating the image with the model and the test data
first_sample = image[0].reshape(14, 14) 
last_sample = image[-1].reshape(14, 14)

average_sample = np.zeros((196,1)) #creating an empty list for the average sample

for i in range(1,1000):
    average_sample += image[i].reshape(-1,1)
average_sample = average_sample * (1/900) #calculating the average of the 900 samples
average_sample = average_sample.reshape(14, 14)    
# Plot results
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(first_sample, cmap='plasma')
axes[0].set_title("First Sample")
axes[1].imshow(last_sample, cmap='plasma')
axes[1].set_title("Last Sample")
axes[2].imshow(average_sample, cmap='plasma')
axes[2].set_title("Average of Last 900 Samples")
plt.show()
#we can observe that first image is very random, the middle image shows more of a shape of a number 5 after the 50 training epochs. I tried this with other numbers of 5, 8 and 4, and it gave the number that I was training my model for,
#The average of the 900 samples gives an shape of a 5, it can be said that this case shows us first couple visible sample configuration is not good
#where this can be related to the burn-in time that is caused by the Gibbs Sampling process. As the visible neuron configuration is generated randomly, it is expected to have the first sets of samples to be 'garbage' or meant to be taken out.
