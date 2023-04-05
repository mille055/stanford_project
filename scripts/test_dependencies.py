import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torchvision
import pydicom
import monai
import pickle
import glob
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import seaborn as sns

from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import time
import os
import copy

import monai
from monai.data import DataLoader, ImageDataset
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, EnsureType


import warnings
warnings.filterwarnings("ignore")

### local imports ###
from config import file_dict, abd_label_dict
from config import column_lists, feats
from config import val_list, train_val_split_percent, random_seed, data_transforms
from config import sentence_encoder, series_description_column, text_label


dum = column_lists['dummies'] 
dirf = file_dict['img_data_dir_local']
features = feats

print(dum)
print(dirf)
print(features)