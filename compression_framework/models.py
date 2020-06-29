import torch
from torch import nn
import torch.nn.functional as F


class ComCNN(nn.Module):
    """
    Construction Convolutional Neural Network(ComCNN), image encoder model of the framework.

    ComCNN consists of 3 layers as per original implementation:
    1. Conv3x3 + ReLU , stride=1, in=3, out=64
    2. Conv3x3 + BatchNorm + ReLU, stride=1, in=64, out=64
    3. Conv3x3, stride=1, in=64, out=3 
    """

    def __init__(self):
        super(ComCNN, self).__init__()
        self.layer_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.layer_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, padding=0),
            nn.BatchNorm2d(64),
        )
        self.layer_3 = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


class RecCNN(nn.Module):
    """
    Reconstruction Convolutional Neural Network(RecCNN), image decoder model of the framework.

    RecCNN consists of 21 layers in total, as per original implementation:
    1. Conv3x3, stride=1, in=3, out=64
    2-19. Conv3x3, stride=1, in=64, out=64
    20. Conv3x3, stride=1, in=64, out=3 
    """

    def __init__(self):
        super(RecCNN, self).__init__()
        self.layer_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.intermediate_layers = []
        for _ in range(18):
            self.intermediate_layers.append(nn.Conv2d(64, 64 ,3))
            self.intermediate_layers.append(nn.BatchNorm2d(64))
            self.intermediate_layers.append(nn.ReLU6())
        self.intermediate_layers = nn.Sequential(*self.intermediate_layers)
        self.final_layer = nn.Conv2d(64, 3, 3)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.intermediate_layers(x)
        x = self.final_layer(x)
        return x