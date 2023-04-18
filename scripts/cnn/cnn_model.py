import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

from config import classes


criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
  

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=len(classes)):
        super(CustomResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)

        # Replace the last fully connected layer to match the number of classes
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)
