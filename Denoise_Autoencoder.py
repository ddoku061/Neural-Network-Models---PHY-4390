#Code by Doga Dokuz, 2025
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset

#choosing the device
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

#importing MNIST dataset, inspired by the assignment 3, question 3
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

#getting two images and their labels from training and test sets
image1_training, label1_training = training_data[0]
image2_training, label2_training = training_data[8]
image1_test, label1_test = test_data[0]
image2_test, label2_test = test_data[8]

#stacking images into tensors for training and testing
training_image_data = torch.stack([image1_training.squeeze(), image2_training.squeeze()]).to(device)
test_image_data = torch.stack([image1_test.squeeze(), image2_test.squeeze()]).to(device)
num_noisy_images = 500
#creating a function to add noise
def create_noise(images, num_noisy_images):
    num_image = len(images)
    noisy_dataset = torch.zeros((num_noisy_images * num_image, 28, 28)).to(device) #creating a empty tensor for noisy images
    clean_dataset = torch.zeros((num_noisy_images * num_image, 28, 28)).to(device) #creating a empty tensor for clean images
    
    for i in range(num_image): #iterating over the images, in case we have just 2 images
        for n in range(num_noisy_images): #adding noise to the images individually
            noise = torch.FloatTensor(28, 28).uniform_(-1, 1).to(device) #creating a noise tensor by adding a random number between -1 and 1
            noisy_dataset[n + i * num_noisy_images] = images[i] + noise #adding noise to the noise tensor
            clean_dataset[n + i * num_noisy_images] = images[i] #keeping the original image in the clean tensor
    
    noisy_dataset = noisy_dataset.clamp(0, 1) #clamping all the noisy values between 0 and 1
    return noisy_dataset, clean_dataset

#creating noisy dataset
noisy_imgs, clean_imgs = create_noise(training_image_data,num_noisy_images) #sending the images through the noise adding function
noisy_imgs = noisy_imgs.view(-1, 28 * 28) #reshaping
clean_imgs = clean_imgs.view(-1, 28 * 28) #reshaping

#creating DataLoader
dataset = TensorDataset(noisy_imgs, clean_imgs) #creating a dataset from the noisy and clean images to reconstruct our images
dataloader = DataLoader(dataset, batch_size=64, shuffle=True) 


#creating Autoencoder Neural Network model
class Autoencode(nn.Module):
    def __init__(self, hidden_size=64):
        super(Autoencode, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, hidden_size), #our input image size is 28x28 to hidden layer
            nn.ReLU() #activation function
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 28*28), #hidden layer to output image size, back to 28x28
            nn.Sigmoid() #activation function
        )
    
    def forward(self, x): #feeding the input through the encoder then decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x


hidden_size = 64
model = Autoencode(hidden_size).to(device) #calling the model and sending it to the device
criterion = nn.MSELoss() #mse loss function
optimizer = optim.Adam(model.parameters(), lr=0.001) #adam optimizer


num_epochs = 500 #number of epochs
#empty list to store our values for train error and # of epochs
epochs = []
training_loss = []

#training our model, similar to assignment 3
def train(dataloader,model):
    model.train() #setting the model to the training mode
    for epoch in range(num_epochs):
        epochs.append(epoch)
        total_loss = 0
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad() #setting the gradients to zero
            outputs = model(noisy) #taking the output of our model with the input image tensor
            loss = criterion(outputs, clean) #computing the loss function
            loss.backward() #backpropagation
            optimizer.step() #updating the weights
            total_loss += loss.item() #transforming the loss function for a readable format
        #printing the loss function every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

        total = total_loss/len(dataloader)
        #storing the training loss and epoch number to plot the loss function later
        training_loss.append(total)
    return training_loss

train(dataloader,model)
print(len(training_loss))

#plotting the loss values over epochs
plt.yscale('log')
plt.title(f'MSE Loss Values over {num_epochs} Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.plot(epochs,training_loss)
plt.grid('True')
plt.show()

#evaluating noisy images
def test():
    model.eval()
    with torch.no_grad():
        recon_imgs = model(noisy_imgs).view(-1, 28, 28).cpu().numpy()
    return recon_imgs
recon_imgs = test()

#plotting our original image, noisy image and denoised image
fig, axes = plt.subplots(3, 2, figsize=(6, 9))
for i in range(2):
    axes[0, i].imshow(noisy_imgs[i * num_noisy_images].view(28,28).cpu(), cmap='gray')
    axes[0, i].set_title("Noisy Input")
    axes[1, i].imshow(clean_imgs[i * num_noisy_images].view(28,28).cpu(), cmap='gray')
    axes[1, i].set_title("Original Image")
    axes[2, i].imshow(recon_imgs[i * num_noisy_images], cmap='gray')
    axes[2, i].set_title("Denoised Output")
plt.tight_layout()
plt.show()
