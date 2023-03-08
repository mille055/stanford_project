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

        
  
