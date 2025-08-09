
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from token import tokenize_c

#for logging data to tensorboard
writer = SummaryWriter("training_runs/mnist_digit_recog")


# define model structure
class NueralNet(nn.Module):
    def __init__(self):
        super().__init__()
        # layer one
        self.layer_1 = nn.Linear(28*28, 128)
        #output layer
        self.layer_2 = nn.Linear(128, 10)


    # make prediciton!
    def forward(self, x):
            #flatten down to one dimension, inferring batch size with -1
        x = x.view(-1, 28*28)

        x = func.silu(self.layer_1(x))
        x = self.layer_2(x)
        return x



def train_model(model, criterion, optimizer, num_epochs):

    writer = SummaryWriter("training_runs/mnist_digit_recog")

    #           no fancy transfroms yet
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST('./training_data', train=True, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)

    for epoch in range(num_epochs):  # Train for x epochs
        
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            
            tokenize_c(code)

            # Forward pass
            outputs = model(images)

            loss = criterion(outputs, labels)

            # clear
            optimizer.zero_grad()
            # gradient
            loss.backward()
            # update
            optimizer.step()

            # this is just for modelling preformance
            if(torch.max(outputs, 1) == labels):
                correct+=1
            total+=1
        # log for tensorboard
        writer.add_scalar("Loss/train", loss.item() / len(train_loader), epoch)
        writer.add_scalar("Accuracy/train", correct / total, epoch)

        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")





def test_model(model, criterion, num_epochs):

    writer = SummaryWriter("test_runs/mnist_digit_recog")

    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST('./test_data', train=False, download=True, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=60, shuffle=True)

    for epoch in range(num_epochs):  # Train for x epochs

        total = 0
        correct = 0

        for batch_idx, (images, labels) in enumerate(test_loader):
        
            # Forward pass
            outputs = model(images)

            loss = criterion(outputs, labels)

            # this is just for modelling preformance
            if(torch.max(outputs, 1) == labels):
                correct+=1
            total+=1

        # log for tensorboard
        writer.add_scalar("Loss/train", loss.item() / len(test_loader), epoch)
        writer.add_scalar("Accuracy/train", correct / total, epoch)

        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")