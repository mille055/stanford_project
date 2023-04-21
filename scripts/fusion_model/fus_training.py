import numpy as np
import pandas as pd
import os
import torch
import pickle

from fusion_model.fus_model import FusionModel
from cnn.cnn_inference import pixel_inference, load_pixel_model
from metadata.meta_inference import get_meta_inference
from NLP.NLP_inference import get_NLP_inference, load_NLP_model
from config import feats_to_keep, classes, model_paths
from model_container import ModelContainer
from utils import *


train_df, val_df, test_df = create_datasets(train_datafile, val_datafile, test_datafile)

#get processed columns for metadata columns
X_train_meta = preprocess(train_df)
X_val_meta = preprocess(val_df)
X_test_meta = preprocess(test_df)
y_train = train_df.label
y_val = val_df.label
y_test = test_df.label

