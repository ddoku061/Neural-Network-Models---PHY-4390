#Code by Doga Dokuz, 2025

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

#defining a convolutional neural network using the LeNet architecture
class LeNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2), #convolutional layer 1, input 32x32, output 28x28x6
            nn.Sigmoid(), #choosing sigmoid as the activation function due to our class notes
            nn.AvgPool2d(kernel_size=2, stride=2),  # convolutional layer 2 for sample pooling, input 28x28x6, output 14x14x6
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  #convolutional layer 3, input 14x14x6, output 10x10x16
            nn.Sigmoid(), #activation function
            nn.AvgPool2d(kernel_size=2, stride=2), #last convolutional layer 4 for sample pooling input 10x10x16, output 5x5x16            
        )
        self.feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120), #feature layer 5 input 400, output 120
            nn.Sigmoid(), #applying sigmoid function in the hidden layers
            nn.Linear(in_features=120, out_features=84), #feature layer 6 input 120, output 84
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10), #feature layer 7, input 84, output 10 because there are 10 different digits, 0 to 9
        )
            
    # Defining the forward pass of the network
    def forward(self, x):     
        return self.feature(self.conv_layer(x))

#choosing our device to run our code in
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
model = LeNetCNN().to(device) #calling our model and sending it to the device
criterion = nn.CrossEntropyLoss() #choosing our loss function as cross entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001) #choosing our optimizer as ADAM with learning rate 0.001

#using ADAM optimizer could give us higher accuracy and faster convergence in a shorter time, so personally I am not sure if it is ideal to run the model for 2000 epochs. 
num_epochs=250
#while writing the code, I have tried epochs and my code ran for 10 hours, so I decided to keep the epoch number short.

#defining our training and testing functions
def train(model, train_loader, optimizer, criterion, num_epochs):
    train_loss_plot = []
    test_loss_plot = []
    
    train_accuracy_plot = []
    test_accuracy_plot = []
    #same old training function from the previous questions
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            #measuring the accuracy of our output
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
        #tried calling the test function here, this might have increased the run time
        test_loss, test_acc= test(model, test_loader, criterion)
        
        #storing train and test losses for plotting
        train_loss_plot.append(train_loss / len(train_loader))
        test_loss_plot.append(test_loss)
        train_acc = 100. * correct / total
        
        train_accuracy_plot.append(train_acc)
        test_accuracy_plot.append(test_acc)
        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Test Loss: {test_loss:.4f}, , Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    return train_loss_plot, test_loss_plot, train_accuracy_plot, test_accuracy_plot

#creating a test function to evaluate the model
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            test_loss += loss.item()
            #measuring the accuracy of our output
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            accuracy = 100 * correct/total
    return test_loss / len(test_loader), accuracy

train_losses, test_losses, train_accuracy , test_accuracy = train(model, train_loader, optimizer, criterion, num_epochs)
#calculating the gap between train and test loss
gap_losses = np.abs(np.array(train_losses) - np.array(test_losses))


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.plot(gap_losses, label="Gap Loss", linestyle="dashed")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(test_accuracy, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.show()
