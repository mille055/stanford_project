import torch
from cnn.cnn_model import CustomResNet50
from cnn.cnn_data_loaders import get_data_loaders
from datetime import datetime

from config import classes


# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the custom model
num_classes = len(classes)
model = CustomResNet50(num_classes)
model = model.to(device)  # Move the model to the appropriate device

# Get the data loaders
batch_size = 64
train_loader, val_loader = get_data_loaders(batch_size)

# Define the training loop
for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)  # Move the input data to the appropriate device