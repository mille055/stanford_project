import pandas as pd
import numpy as np
import pydicom
from pydicom.dataset import Dataset as DcmDataset
from pydicom.tag import BaseTag as DcmTag
from pydicom.multival import MultiValue as DcmMultiValue


### local imports ###
from config import config_dict


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
  
 ### takes the contents of the dataframe column with path/filenames and converts pieces into separate df columns ###   
 def expand_filename_into_columns(df, cols):
    for iterator in range(len(cols)):
        df[cols[iterator]] = df['fname'].astype(str).apply(lambda x: x.split('/')[iterator])
    return df

        
  
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
        source = '/volumes/cm7/Abdominal_MRI_dataset_split/'
        dest = '/content/gdrive/MyDrive/WW_MRI_abd2/split/'

        img_file = self.data_df.file_info[idx]
        rel = os.path.relpath(img_file, source)
        img_file_new = os.path.join(dest,rel)
        
        #print('getting file', img_file)
        ds = pydicom.dcmread(img_file_new)
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