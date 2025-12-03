"""
Docstring for src.model
In this module, i am making three varieties

1. a super simple custom CNN 
3. an advanced custom CNN 
3. an advance model from a pretrained model 
"""

from torch import nn 
from torch.nn import functional as F
import torch

from torch import Tensor

# the super simple CNN-based model 

class SimpleTomatoCNN(nn.Module): 

    # te object constructor 
    def __init__(self, n_classes: int): 
        super().__init__()

        # convolution layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # pooling layers
        self.pool = nn.MaxPool2d(2, 2)

        # fully connected layers
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, n_classes)

    
    # overwriting the forward method
    def forward(self, x: Tensor): 
        # passing x through the convolutions
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # flattening the tensor x
        x = x.view(x.size(0), -1)

        # passing x through the linear layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



