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

from torchvision.models import mobilenet_v3_small
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


# the advanced CNN class 
class AdvancedTomatoCNN(nn.Module):
    def __init__(self, n_classes=10): 
        super().__init__()


        # making a function to build each convolution block
        def block(in_c, out_c): 
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), 
                nn.BatchNorm2d(out_c), 
                nn.ReLU(), 
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1), 
                nn.BatchNorm2d(out_c), 
                nn.ReLU(), 
                nn.MaxPool2d(2, 2)
            )
        

        # using the convolution block to make multiple convolution layers
        self.layer1 = block(3, 32)
        self.layer2 = block(32, 64)
        self.layer3 = block(64, 128)
        self.layer4 = block(128, 256)

        # makeing the linear layers and the dropout layers
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, n_classes)


    def forward(self, x: torch.Tensor): 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # flattening x 
        x = x.view(x.size(0), -1)

        # passing x through the linear layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x



class SimplePretrainedTomatoMobileNet(nn.Module): 
    def __init__(self, n_classes): 
        super().__init__()

        self.model = mobilenet_v3_small(pretrained=True)

        # freezing the backbone of the pretrained model
        for param in self.model.features.parameters(): 
            param.requires_grad = False

        # to replace the classifier
        in_features = self.model.classifier[3].in_features 
        self.model.classifier[3] = nn.Linear(in_features=in_features, out_features=n_classes)


    def forward(self, x): 
        return self.model(x)

