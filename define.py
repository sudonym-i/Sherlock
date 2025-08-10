import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as func


# define model structure
class NueralNet(nn.Module):
    def __init__(self, dimensionality):
        super().__init__()
        # layer one
        self.layer_1 = nn.Linear(dimensionality, dimensionality/2)
        #output layer
        self.layer_2 = nn.Linear(dimensionality/2, 10)


    # make prediciton!
    def forward(self, x):
            #flatten down to one dimension, inferring batch size with -1
        x = x.view(-1, 28*28)

        x = func.silu(self.layer_1(x))
        x = self.layer_2(x)
        return x


