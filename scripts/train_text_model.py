import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk, spacy
import sklearn
from sklearn.model_selection import train_test_split
import os
import os.path
import glob
from __future__ import print_function
import sys
from random import shuffle
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import time
import os
import copy

from config import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_text_data()
    data_dir = txt_data_dir
    #train_df = pd.read_pickle(data_dir+'train_img_df.pkl')
    #test_df = pd.read_pickle(data_dir+'test_img_df.pkl')
    train_csv = pd.read_csv(data_dir + 'trainfiles.csv')
    test_csv = pd.read_csv(data_dir + 'testfiles.csv')

    #run once at start to rid unneccesary column
    train_csv.drop('Unnamed: 0', axis=1, inplace=True)
    test_csv.drop('Unnamed: 0', axis=1, inplace=True)

    # create shortened dataframes for train and test
    train_csv_short = shorten_df(train_csv, selection_fraction = 0.5)
    test_csv_short = shorten_df(test_csv, selection_fraction = 0.5)

    return train_csv_short, test_csv_short

