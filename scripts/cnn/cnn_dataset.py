import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom

from config import file_dict, classes


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
        # rel = os.path.relpath(img_file, source)
        # img_file = os.path.join(dest,rel)
        
        #print('getting file', img_file)
        ds = pydicom.dcmread(img_file)
        img = np.array(ds.pixel_array, dtype=np.float32)
        #img = img/255.
        #img = cv2.resize(img, (224,224))
        img = img[np.newaxis]
        img = torch.from_numpy(np.asarray(img))
        
        if self.transform:
            img = self.transform(img)
        #print('after transform', img.dtype, img.shape)
            
        x = img
        labl = self.data_df.label[idx]
        adjusted_label = classes.index(labl)
        y = torch.tensor(adjusted_label, dtype=torch.long)  # Use torch.long instead of torch.float32 
        
        return (x,y)