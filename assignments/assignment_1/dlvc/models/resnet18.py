import torch
from torch import nn
import torchvision.models as models

class ResNet18Dropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ResNet18Dropout, self).__init__()
        # Load the pretrained ResNet18 model
        self.resnet18 = models.resnet18(pretrained=True)
        
        # Get the input dimension of the last layer
        num_ftrs = self.resnet18.fc.in_features
        
        # Re-define the last layer (fc) to include dropout
        self.resnet18.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        return self.resnet18(x)