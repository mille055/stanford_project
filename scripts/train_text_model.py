from __future__ import print_function
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk, spacy
import sklearn
from sklearn.model_selection import train_test_split
import os
import os.path
import glob

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
from datetime import datetime
import pickle 

from config import *
from utils import shorten_df, plot_and_save_cm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
senttrans_model = SentenceTransformer(sentence_encoder, device=device)


def load_text_data(train_df, test_df):
    
    # train_csv = pd.read_csv(train_csv_file)
    # test_csv = pd.read_csv(test_csv_file)
    
    # #run once at start to rid unneccesary column
    # train_csv.drop('Unnamed: 0', axis=1, inplace=True)
    # test_csv.drop('Unnamed: 0', axis=1, inplace=True)

    # create shortened dataframes for train and test
    train_df_short = shorten_df(train_df, selection_fraction = 0.5)
    test_df_short = shorten_df(test_df, selection_fraction = 0.5)

    # create train, val, test datasets
    val = train_df_short[train_df_short.patientID.isin(val_list)].reset_index(drop=True)
    train = train_df_short[~train_df_short.index.isin(val.index)].reset_index(drop=True)
    test = test_df_short.reset_index(drop=True)

    # train = train_df.reset_index(drop=True)
    # val = val_df.reset_index(drop=True)
    # test = test_df.reset_index(drop=True)
    print(train.columns)

    X_train = train[series_description_column]
    y_train = train[text_label]

    X_val = val[series_description_column]
    y_val = val[text_label]

    X_test = test[series_description_column]
    y_test = test[text_label]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_text_model(senttrans_model=senttrans_model, train_df=train_data, test_df=test_data):
    X_train, y_train, X_val, y_val, X_test, y_test = load_text_data(train_df, test_df)
    
    #encode the text labels in the train, val, and test datasets
    X_train_encoded = [senttrans_model.encode(doc) for doc in X_train.to_list()]
    X_val_encoded = [senttrans_model.encode(doc) for doc in X_train.to_list()]
    X_test_encoded = [senttrans_model.encode(doc) for doc in X_test.to_list()]

    # Train a classification model using logistic regression classifier
    logreg_model = LogisticRegression(solver='saga')
    logreg_model.fit(X_train_encoded, y_train)
    preds = logreg_model.predict(X_train_encoded)
    acc = sum(preds == y_train) / len(y_train)
    print('Accuracy on the training set is {:.3f}'.format(acc))

    ## assess on the val set
    preds_val = logreg_model.predict(X_val_encoded)
    acc_val = sum(preds_val == y_val)/ len(y_val)
    ## display results on test set
    print('Accuracy on the val set is {:.3f}'.format(acc_val))

    ## assess on the test set
    preds_test = logreg_model.predict(X_test_encoded)
    acc_test = sum(preds_test == y_test) / len(y_test)
    ## display results on test set
    print('Accuracy on the test set is {:.3f}'.format(acc_test))


    #export model
    txt_model_filename = "../models/text_model"+ datetime.now().strftime('%Y%m%d') + ".st"
    pickle.dump(logreg_model, open(txt_model_filename, 'wb'))

    return preds, acc, preds_val, acc_val, preds_test, acc_test, logreg_model




def list_incorrect_text_predictions(ytrue, ypreds):
    ytrue = ytrue.to_list()
    ypreds = ypreds.to_list()
    ylist = zip(ytrue, ypreds)

    y_incorrect_list = [x for x in ylist if x[0]!=x[1]]

    return y_incorrect_list

## test
## pickled dataframes
test_data = pd.read_pickle('../data/X_test02282023.pkl')
train_data = pd.read_pickle('../data/X_train02282023.pkl')
print(train_data.columns)

preds, acc, preds_val, acc_val, preds_test, acc_test, logreg_model = train_text_model()
list = list_incorrect_text_predictions(y_test, preds_test)
print(list)