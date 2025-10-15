#Code by Doga Dokuz, 2025

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

#I found another way to import MNIST dataset online
transform = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

#selecting two different images and their labels manually
img1, label1 = mnist_data[0]  # First image
img2, label2 = mnist_data[9]  # Second image

#stacking images into a tensor of shape (2, 28, 28)
img_tensor = torch.stack([img1.squeeze(), img2.squeeze()])

#plotting the images we retrieved from the mnist dataset
fig, axes = plt.subplots(1, 2, figsize=(6, 9))
axes[0].imshow(img_tensor[0], cmap="gray")
axes[0].set_title(f'Digit {label1}')
axes[1].imshow(img_tensor[1],cmap="gray")
axes[1].set_title(f'Digit {label2}')
plt.tight_layout()
plt.show()
#creating our neural network for autoencoding
class Autoencoder(nn.Module):
    def __init__(self, hidden_size):
        super(Autoencoder, self).__init__() #calling the parent class constructor
        self.encoder = nn.Sequential( #creating layers of our encoder with linear and relu activation function
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential( #creating layers of our decoder with linear and sigmoid functions
            nn.Linear(hidden_size, 28 * 28),
            nn.Sigmoid()
        )
    
    def forward(self, x): #creating a forward function to pass the input through the encoder then decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, 1, 28, 28)


hidden_size = 15  
# the width of the hidden layer should be at least 15, after trial and error, I found that 15 is the best value for the hidden layer width, 
# although we tried different hidden layers, 15 had the smoothiest convergence
num_epochs = 500 #many epochs are needed to achieve the convergence of loss function in our model
learning_rate = 0.01 #we picked our learning rate to be 0.01 after trial and error


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #finding a device for our model to run through
model = Autoencoder(hidden_size).to(device) #calling our model and sending it to the device
criterion = nn.MSELoss() #choosing mean squared error as our loss function 
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #we choose Adam as our optimizer

#preparing the input tensor for our model
img_tensor = img_tensor.unsqueeze(1).to(device)  # Reshape to (2, 1, 28, 28)

training_plot = []
epochs = []
#starting the training process of our model
def train():
    model.train() #setting the model to the training mode
    for epoch in range(num_epochs):
        optimizer.zero_grad() #setting the gradients to zero
        output = model(img_tensor) #taking the output of our model with the input image tensor
        loss = criterion(output, img_tensor) #computing the loss function of mse
        loss.backward() #backpropagating
        optimizer.step() #updating the weights
        train_loss = loss.item() #transforming the loss function for a readable format
        #storing the training loss and epoch number to plot the loss function later
        training_plot.append(train_loss)
        epochs.append(epoch)
        #printing the loss function every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
    return training_plot, epochs
    
training_plot, epochs = train()

#reconstructing images with autoencoding model
def test():
    model.eval() #setting the model to evaluation (test) mode
    with torch.no_grad():
        reconstructed = model(img_tensor) #passing the input tensor through the model
    return reconstructed

reconstructed = test()

#plotting our loss function over epochs
plt.title(f'MSE Loss Values over {num_epochs} Epochs')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid('True')
plt.plot(epochs,training_plot)

#plotting original and reconstructed images from the dataset
fig, axes = plt.subplots(2, 2, figsize=(6, 6))
for i in range(2):
    axes[i, 0].imshow(img_tensor[i].cpu().squeeze(), cmap='gray')
    axes[i, 0].set_title("Original")
    axes[i, 1].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
    axes[i, 1].set_title("Reconstructed")


plt.show()
