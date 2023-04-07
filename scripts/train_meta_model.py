import numpy as np
import pydicom
from sklearn.model_selection import KFold, GroupKFold, cross_val_score, cross_validate, GroupShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, plot_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt  
from sklearn.metrics import precision_recall_fscore_support as score
import pprint
import pickle

from config import file_dict, feats, column_lists
from config import abd_label_dict, val_list, train_val_split_percent, random_seed, data_transforms
from config import sentence_encoder, series_description_column, text_label
from utils import *

#grid search for hyperparameters
def train_fit_parameter_trial(train, y, features, fname='model-run.skl'):
    "Train a Random Forest classifier on `train[features]` and `y`, then save to `fname` and return."
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf.fit(train[features], y)
    print('Parameters currently in use:\n')
    pprint(clf.get_params())


    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 20, stop = 500, num = 20)]
# Number of features to consider at every split
    max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 660, num = 10)]
    max_depth.append(None)
# Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 20]
# Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4, 8]
# Method of selecting samples for training each tree
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    
    clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)
    clf_random.fit(train[features], y)
    opt_clf = clf_random.best_estimator_
    pprint(clf_random.best_params_)
    pickle.dump(opt_clf, open(fname, 'wb'))
    #dump(clf_random, fname)
    return opt_clf

