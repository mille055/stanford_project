import pandas as pd
import numpy as np
import pydicom

from pydicom.dataset import Dataset as DcmDataset
from pydicom.tag import BaseTag as DcmTag
from pydicom.multival import MultiValue as DcmMultiValue
from pathlib import Path
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import os, sys, glob, re
import numpy as np, pandas as pd
from joblib import dump, load


### local imports ###
from config import file_dict, abd_label_dict
from config import column_lists, feats
from config import val_list, train_val_split_percent, random_seed, data_transforms
from config import sentence_encoder, series_description_column, text_label


### gets the dicom files from a provided directory ###
def get_dicoms(path, first_dcm=False, **kwargs):
    "Walk `path` to get DICOM file names from specific extensions, then read files into a `pandas.DataFrame`. If `first_dcm=True`, only read first file from each folder."
    fns = L()
    extension_list=['.dcm','.dicom','.dcim','.ima']
    print("Finding DICOM files. This may take a few minutes.")
    if first_dcm:
        for r, d, f in os.walk(path):
            if f:
                if Path(f[0]).suffix.lower() in extension_list:
                    fns.append(Path(f'{r}/{f[0]}'))
    else:
        fns = L()
        for r, d, fs in os.walk(path):
            for f in fs:
                if Path(f).suffix.lower() in extension_list:
                    fns.append(Path(f'{r}/{f}'))
    print("Reading DICOM files with extensions .dcm, .dicom, .dcim, or .ima. This may take a few minutes, depending on the number of files to read...")
    df = pd.DataFrame.from_dicoms(fns, **kwargs)
    return fns, df

### Reads a DICOM file and returns the corresponding pydicom.Dataset ###
def dcmread(fn: Path, no_pixels=True, force=True):
    return pydicom.dcmread(str(fn), stop_before_pixels=no_pixels, force=force)


def as_dict(self: DcmDataset, filt=True, split_multi=False):
    if filt:
        vals = [self[o] for o in self.keys() if self[o].keyword in _cols]
    else:
        vals = [self[o] for o in self.keys()]
    items = [(v.keyword, v.value.name) if v.keyword == 'SOPClassUID' else (v.keyword, v.value) for v in vals]
    res = dict(items)
    res['fname'] = self.filename
    if split_multi:
        for k, v in items: _split_elem(res, k, v)
        for k in res: res[k] = _cast_dicom_special(res[k])
    return res


def _dcm2dict(fn, excl_private=False, **kwargs):
    ds = fn.dcmread(**kwargs)
    if excl_private: ds.remove_private_tags()
    return ds.as_dict(**kwargs)


def _from_dicoms(cls, fns, n_workers=0, **kwargs):
    return pd.DataFrame(parallel(_dcm2dict, fns, n_workers=n_workers, **kwargs))
pd.DataFrame.from_dicoms = classmethod(_from_dicoms)


def get_series_fp(fn): return Path(fn).parent

 ### takes the contents of the dataframe column with path/filenames and converts pieces into separate df columns ###   
def expand_filename_into_columns(df, cols):
    for iterator in range(len(cols)):
        df[cols[iterator]] = df['fname'].astype(str).apply(lambda x: x.split('/')[iterator])
    return df

def compute_plane(row):
    '''
    Computes the plane of imaging from the direction cosines provided in the `ImageOrientationPatient` field.
    The format of the values in this field is: `[x1, y1, z1, x2, y2, z2]`,
    which correspond to the direction cosines for the first row and column of the image pixel data.
    '''
    planes = ['sag', 'cor', 'ax']
    if 'ImageOrientationPatient1' in row.keys():
        dircos = [v for k,v in row.items() if 'ImageOrientationPatient' in k]
    else:
        dircos = row['ImageOrientationPatient']
    dircos = np.array(dircos).reshape(2,3)
    pnorm = abs(np.cross(dircos[0], dircos[1]))
    return planes[np.argmax(pnorm)]

#_re_extra_info = re.compile(r'[<\([].*?[\]\)>]')

def rm_extra_info(t):
    "Remove extraneous info in closures"
    return re.compile(r'[<\([].*?[\]\)>]').sub('', t).strip()


def detect_contrast(row):
    sd = rm_extra_info(str(row['SeriesDescription']).lower())
    if _c.search(sd): return 1
    c = row['ContrastBolusAgent']
    if type(c) == str: return 1
    return 0

def _find_seq(sd):
    if _t1.search(sd):
        if _spgr.search(sd): return 'spgr'
        if _t1_in.search(sd): return 'in phase'
        if _t1_out.search(sd): return 'opposed phase'
        if _water.search(sd): return 'dixon water'
        if _fat.search(sd): return 'dixon fat'    
        if _pv.search(sd): return 'portal venous'
        if _eq.search(sd): return 'equilibrium'
        if _art.search(sd): return 'early dynamic'
        if _delayed.search(sd): return 'hepatobiliary'
        else: return 't1'
    if _t1_in.search(sd): return 'in phase'
    if _t1_out.search(sd): return 'opposed phase'
    if _water.search(sd): return 'dixon water'
    if _fat.search(sd): return 'dixon fat'    
    if _pv.search(sd): return 'portal venous'
    if _eq.search(sd): return 'equilibrium'
    if _art.search(sd): return 'early dynamic'
    if _delayed.search(sd): return 'hepatobiliary'
#    if _spgr.search(sd): return 'spgr'
    if _t2.search(sd):
        if _flair.search(sd): return 'flair'
        elif _swi.search(sd): return 'swi'
        else: return 't2'
    if _flair.search(sd): return 'flair'
    if _swi.search(sd): return 'swi'
    if _dwi.search(sd): return 'dwi'
    if _adc.search(sd): return 'dwi'
    if _eadc.search(sd): return 'dwi'
    if _mra.search(sd): return 'mra'
    if _loc.search(sd): return 'loc'
    if _other.search(sd): return 'other'
    return 'unknown'

def _extract_label(sd):
    t = rm_extra_info(str(sd).lower())
    return _find_seq(t)

def extract_labels(df):
    "Extract candidate labels from Series Descriptions and computed plane"
    df1 = df[['fname', 'SeriesDescription']].copy()
    df1['fname'] = df1.fname.apply(get_series_fp)
    print("Computing planes of imaging from `ImageOrientationPatient`.")
    df1['plane'] = df.apply(compute_plane, axis=1)
    print("Extracting candidate labels from `SeriesDescription`.")
    df1['seq_label'] = df1['SeriesDescription'].apply(_extract_label)
    print("Detecting contrast from `SeriesDescription` and `ContrastMediaAgent`.")
    df1['contrast'] = df.apply(detect_contrast, axis=1)
    return df1


def _make_col_binary(df, col):
    s = df[col].isna()
    if any(s):
        df[col] = s.apply(lambda x: 0 if x else 1)
    else:
        targ = df.loc[0, col]
        df[col] = df[col].apply(lambda x: 0 if x == targ else 1)

def make_binary_cols(df, cols):
    df1 = df.copy()
    for col in cols:
        if col in df.columns:
            _make_col_binary(df1, col)
        else: 
            df1[col]=0
            _make_col_binary(df1,col)
    return df1

def rescale_cols(df, cols):
    df1 = df.copy()
    scaler = MinMaxScaler()
    df1[cols] = scaler.fit_transform(df1[cols])
    return df1.fillna(0)

def get_dummies(df, cols=column_lists['dummies'], prefix=column_lists['d_prefixes']):
    df1 = df.copy()
    for i, col in enumerate(cols):
        df1[col] = df1[col].fillna('NONE')
        mlb = MultiLabelBinarizer()
        df1 = df1.join(
            pd.DataFrame(mlb.fit_transform(df1.pop(col)), columns=mlb.classes_).add_prefix(f'{prefix[i]}_')
        )
    return df1

def preprocess(df, keep= column_lists['keep'], dummies= column_lists['dummies'], d_prefixes= column_lists['d_prefixes'], binarize= column_lists['binarize'], rescale= column_lists['rescale']):
    "Preprocess metadata for Random Forest classifier to predict sequence type"
    print("Preprocessing metadata for Random Forest classifier.")
    df1 = exclude_other(df)
    print(f"Have received {df1.shape[0]} entries.")
    df1 = df1[[col for col in keep if col in df1.columns]]
    if df1['PixelSpacing'].any:
        df1['PixelSpacing'] = df1['PixelSpacing'].apply(lambda x: x[0])
    df1 = get_dummies(df1, dummies, d_prefixes)
    df1 = make_binary_cols(df1, binarize)
    df1 = rescale_cols(df1, rescale)
    for f in _features:
        if f not in df1.columns:
            df1[f] = 0
    return df1

#get labels from a text file in the format of comma deliminted four columns ()
def labels_from_file(label_path, column_names):

    label_df = pd.read_csv(label_path,header=None)
    label_df.columns=column_names

    
    return label_df


def convert_labels_from_file(label_df):
    labels=label_df.copy()
    labels['GT label'] = labels['label_code'].astype(str).apply(lambda x: abd_label_dict[x]['short'])
    labels['GT plane'] = labels['label_code'].astype(str).apply(lambda x: abd_label_dict[x]['plane'])
    labels['GT contrast'] = labels['label_code'].astype(str).apply(lambda x: abd_label_dict[x]['contrast'])
    labels['patientID'] = labels['patientID'].astype(str)
#    labels['Parent_folder'] = labels['fname'].astype(str).apply(lambda x: x.split('/')[0])
#    labels['patientID'] = labels['fname'].astype(str).apply(lambda x: x.split('/')[1]).astype(int)
#    labels['exam'] = labels['fname'].astype(str).apply(lambda x: x.split('/')[2])
#    labels['series'] = labels['fname'].astype(str).apply(lambda x: x.split('/')[3])
    
    return labels

def expand_filename_into_columns(df, cols):
    for iterator in range(len(cols)):
        df[cols[iterator]] = df['fname'].astype(str).apply(lambda x: x.split('/')[iterator])
    return df

def train_setup(df, preproc=True):
    "Extract labels for training data and return 'unknown' as test set"
    if preproc:
        df1 = preprocess(df)
        labels = extract_labels(df1)
        df1 = df1.join(labels[['plane', 'contrast', 'seq_label']])
    else:
        df1 = df.copy()
    filt = df1['seq_label'] == 'unknown'
    train = df1[~filt].copy().reset_index(drop=True)
    test = df1[filt].copy().reset_index(drop=True)
    y, y_names = pd.factorize(train['seq_label'])
    return train, test, y, y_names

def train_setup_abdomen(df, cols=['patientID','exam','series'], preproc=False, need_labels=False):

    if preproc:
        df1=preprocess(df)
        
    else:
        df1=df.copy()
    
    if need_labels:

        labels = extract_labels(df1)
        df1 = df1.merge(labels, on=cols)
 
    length = df1.shape[0]

    #gkf = GroupKFold(n_splits=5)
    #for train_set, val_set in gkf.split(df1, groups=df1['patientID']):
    #    train, val = df1.loc[train_set], df1.loc[val_set]
   
    train_set, val_set = next(GroupShuffleSplit(test_size=.20, n_splits=1, random_state = 42).split(df1, groups=df1['patientID']))

    train = df1.iloc[train_set]
    val = df1.iloc[val_set]
    y, y_names = train['label_code'],train['GT label']
 
    return train, val, y, y_names

# def _get_meta_preds(clf, df, features, y_names=_y_names):
#     y_pred = clf.predict(df[features])
#     y_prob = clf.predict_proba(df[features])
#     preds = pd.Series(y_names.take(y_pred))
#     probas = pd.Series([y_prob[i][pred] for i, pred in enumerate(y_pred)])
#     return pd.DataFrame({'seq_pred': preds, 'pred_proba': probas})

# def predict_from_df(df, features=_features, thresh=0.8, model_path=_model_path, clf=None, **kwargs):
#     "Predict series from `df[features]` at confidence threshold `p >= thresh`"
#     if 'plane' not in df.columns:
#         df1 = preprocess(df)
#         labels = extract_labels(df1)
#         df1 = df1.join(labels[['plane', 'contrast', 'seq_label']])
#     else:
#         df1 = df.copy()
#     if clf:
#         model_path = None
#     else:
#         clf = load(model_path)    
#     df1 = df1.join(_get_preds(clf, df1, features, **kwargs))
#     filt = df1['pred_proba'] < thresh
#     df1['seq_pred'][filt] = 'unknown'
#     return df1

# def predict_from_folder(path, **kwargs):
#     "Read DICOMs into a `pandas.DataFrame` from `path` then predict series"
#     _, df = get_dicoms(path)
#     return predict_from_df(df, **kwargs)

# class Finder():
#     "A class for finding DICOM files of a specified sequence type from a specific ."
#     def __init__(self, path):
#         self.root = path
#         self.fns, self.dicoms = get_dicoms(self.root)
#         self.dicoms = preprocess(self.dicoms)
#         self.labels = extract_labels(self.dicoms)
#         self.dicoms = self.dicoms.join(self.labels[['plane', 'contrast']])
        
#     def predict(self,  model_path=_model_path, features=_features, ynames=_y_names, **kwargs):
#         try:
#             self.clf = load(model_path)
#         except FileNotFoundError as e:
#             print("No model found. Try again by passing the `model_path` keyword argument.")
#             raise
#         self.features = features
#         self.ynames = ynames
#         preds = self.clf.predict(self.dicoms[features])
#         self.preds = ynames.take(preds)
#         self.probas = self.clf.predict_proba(self.dicoms[features])
        
#     def find(self, plane='ax', seq='t1', contrast=True, thresh=0.8, **kwargs):
#         try:
#             self.probas
#         except AttributeError:
#             print("Prediction not yet performed. Please run `Finder.predict()` and try again.")
#             raise
#         preds = np.argwhere(self.probas > 0.8)
#         ind = preds[:, 0]
#         pred_names = _y_names.take(preds[:, 1])
#         df = pd.DataFrame(pred_names, index=ind, columns=['seq_pred'])
#         df = self.dicoms[_output_columns].join(df)
#         return df.query(f'plane == "{plane}" and seq_pred == "{seq}" and contrast == {int(contrast)}')
    


def exclude_other(df):
    if 'BodyPartExamined' not in df.columns: return df
    other = ['SPINE', 'CSPINE', 'PELVIS', 'PROSTATE']
    filt = df.BodyPartExamined.isin(other)
    df1 = df[~filt].copy().reset_index(drop=True)
    filt1 = df1.SeriesDescription.str.contains(r'(cervical|thoracic|lumbar)', case=False, na=False)
    df2 = df1[~filt1].reset_index(drop=True)
    filt2 = df2.SOPClassUID == "MR Image Storage"
    return df2[filt2].reset_index(drop=True)  

def load_pickled_dataset(train_file, test_file):
  with open(train_file, 'rb') as f:
    train_df = pickle.load(f)
  with open(test_file, 'rb') as g:
    test_df = pickle.load(g)

  return train_df, test_df

def load_csv_dataset(train_file, test_file, val = True, val_lists = None):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    train_df.drop('Unnamed: 0', axis=1, inplace=True)
    test_df.drop('Unnamed: 0', axis=1, inplace=True)
    if val:
        if val_lists:
            val_df = train_df[train_df.patientID.isin(val_lists)]
            train_df = train_df[~train_df.index.isin(val_df.index)] 
        else:
            train_set, val_set = next(GroupShuffleSplit(test_size=.20, n_splits=1, random_state = 42).split(train_df, groups=train_df['patientID']))
            train_df, val_df = train_set, val_set
        return train_df, val_df, test_df

    else: 
        return train_df, test_df

# this function will select the image in the middle of each series of images, so that only a single image from each series is selected for training
# There is one image from each series for each patient
def shorten_df(df, selection_fraction = 0.5):
  df1 = df.copy()
  grouped_df = df.groupby(['patientID', 'series'])
  sorted_df = grouped_df['file_info'].apply(lambda x: x.sort_values())
  
  selected_filename = grouped_df['file_info'].apply(lambda x: x.sort_values().iloc[int(len(x)*selection_fraction)])

  
  selected_filename = selected_filename.reset_index()
  
  # perform merge and deal with duplicate/unnecessary columns
  df1 = df1.merge(selected_filename, on=['patientID', 'series'], how='left') 
  df_short = df1.drop(['file_info_x', 'img_num'], axis=1)
  df_short = df_short.rename(columns = {'file_info_y': 'file_info'})
  df_short.drop_duplicates(inplace=True)
  df_short.reset_index(drop=True, inplace=True)
  return df_short


  # custom datasest - gets the image data using pydicom.dcmread and transforms
# also gets label from the label column and merges classes 2-5 which are all flavors
# arterial into a single 'arterial' label as label 2

class ImgDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data_df = df
        self.datafileslist = df.file_info
        self.labels = df.label
        self.transform = transform
        
        
    def __len__(self):
        return self.data_df.shape[0]
    
    def __getitem__(self, idx):
        source = file_dict['img_data_dir_local']
        dest = file_dict['img_data_dir_colab']

        img_file = self.data_df.file_info[idx]
        #if in colab, changing path
        #rel = os.path.relpath(img_file, source)
        #img_file_new = os.path.join(dest,rel)
        
        #print('getting file', img_file)
        ds = pydicom.dcmread(img_file)
        img = np.array(ds.pixel_array, dtype=np.float32)
        #img = img/255.
        #img = cv2.resize(img, (224,224))
        img = img[np.newaxis]
        img = torch.from_numpy(np.asarray(img))
        
        #print(img.dtype, img.shape)
        
        
        if self.transform:
            img = self.transform(img)
        #print('after transform', img.dtype, img.shape)
            
        x = img
        labl = self.data_df.label[idx]
      
        # pool the arterial phase into a single label
        if labl in [2,3,4,5]: 
          labl=2
        y = torch.tensor(labl, dtype = torch.float32)
        #print(x,y)
        return (x,y)
        
#visualization of a batch of images
def imshow(img, title):
    img = torchvision.utils.make_grid(img, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)

    plt.axis('off')
    plt.show()

# produce high quality confusion matrix image and save it
# still need to add labels
def plot_and_save_cm(ytrue, ypreds):
    cm = confusion_matrix(ytrue, ypreds)
    plt.figure(figsize=(25, 25))
    plt.tight_layout()
    ConfusionMatrixDisplay(cm).plot(cmap='Blues')
    plt.savefig("./assets/images/ConfusionMatrixSentences" + datetime.now().strftime('%Y%m%d') + ".png", dpi=300, bbox_inches='tight')

    plt.show()

    return plt

