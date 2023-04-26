import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

from app.scripts.config import classes
 
# custom resnet50 model with the top now being a fully connected layer with outputs corresponding to the 19 classes
class CustomResNet50(nn.Module):
    def __init__(self, num_classes=len(classes)):
        super(CustomResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)

        # Replace the last fully connected layer to match the number of classes
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)
