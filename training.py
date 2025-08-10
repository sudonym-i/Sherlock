
import torch
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter\

#for logging data to tensorboard
writer = SummaryWriter("training_runs/mnist_digit_recog")


def train_model(model, criterion, optimizer, num_epochs):

    writer = SummaryWriter("training_runs/mnist_digit_recog")

    train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)

    for epoch in range(num_epochs):  # Train for x epochs
        
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            
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



