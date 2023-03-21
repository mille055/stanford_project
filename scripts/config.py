###### For configuration of the training or inference of the models ######

config_dict = {
  ### locations of assets ###
  data_dir: '/content/gdrive/MyDrive/WW_MRI_abd2/split/',
  dataset_file: './stored_assets/dataset.pkl',
  train_csv = pd.read_csv(data_dir + 'trainfiles.csv'),
  test_csv = pd.read_csv(data_dir + 'testfiles.csv'),
  metadata_model_file: './stored_assets/metadata_model.pkl',
  pixel_model_file: './stored_assets/pixel_model_file.pkl,
  series_description_model_file: './stored_assets/series_description_model_file.pkl',
  val_list = [41, 84, 14, 25, 76, 47,62,0,55,63,101,18,81,3,4,95,66], #using same train/val/test split as in the original split based on the metadata classifier
  random_seed = 42,
  train_val_split_percent = 0.2,

  ### converts numeric labels to textual descriptors ###
  abd_label_dict: {
    '1': {
        'long': 'Anythingelse',
        'short': 'other',
        'plane': 'other',
        'contrast': 'other'
    },
    '2': {
        'long': 'Arterial T1w',
        'short': 'arterial',
        'plane': 'ax',
        'contrast': '1'
    },
    '3': {
        'long': 'Early Arterial T1w',
        'short': 'early_arterial',
        'plane': 'ax',
        'contrast': '1'
    },
    '4': {
        'long': 'Late Arterial T1w',
        'short': 'late_arterial',
        'plane': 'ax',
        'contrast': '1'
    },
    '5': {
        'long': 'Arterial Subtraction',
        'short': 'arterial_sub',
        'plane': 'ax',
        'contrast': '1'
    },
    '6': {
        'long': 'Coronal Late Dynamic T1w',
        'short': 'dynamic_late',
        'plane': 'cor',
        'contrast': '1'
    },
    '7': {
        'long': 'Coronal T2w',
        'short': 't2',
        'plane': 'cor',
        'contrast': '0'
    },
    '8': {
        'long': 'Axial DWI',
        'short': 'dwi',
        'plane': 'ax',
        'contrast': '0'
    },
    '9': {
        'long': 'Axial T2w',
        'short': 't2',
        'plane': 'ax',
        'contrast': '0'
    },
    '10': {
        'long': 'Coronal DWI',
        'short': 'dwi',
        'plane': 'cor',
        'contrast': '0'
    },
    '11': {
        'long': 'Fat Only',
        'short': 'dixon_fat',
        'plane': 'ax',
        'contrast': '0'
    },
    '12': {
        'long': 'Axial Transitional_Hepatocyte T1w',
        'short': 'hepatobiliary',
        'plane': 'ax',
        'contrast': '1'
    },
    '13': {
        'long': 'Coronal Transitional_Hepatocyte T1w',
        'short': 'hepatobiliary',
        'plane': 'cor',
        'contrast': '1'
    },
    '14': {
        'long': 'Axial In Phase',
        'short': 'in_phase',
        'plane': 'ax',
        'contrast': '0'
    },
    '15': {
        'long': 'Coronal In Phase',
        'short': 'in_phase',
        'plane': 'cor',
        'contrast': '0'
    },
    '16': {
        'long': 'Axial Late Dyanmic T1w',
        'short': 'dynamic_equilibrium',
        'plane': 'ax',
        'contrast': '1'
    },
    '17': {
        'long': 'Localizers',
        'short': 'loc',
        'plane': 'unknown',
        'contrast': '0'
    },
    '18': {
        'long': 'MRCP',
        'short': 'mrcp',
        'plane': 'cor',
        'contrast': '0'
    },
    '19': {
        'long': 'Axial Opposed Phase',
        'short': 'opposed_phase',
        'plane': 'ax',
        'contrast': '0'
    },
    '20': {
        'long': 'Coronal Opposed Phase',
        'short': 'opposed_phase',
        'plane': 'cor',
        'contrast': '0'
    },
    '21': {
        'long': 'Proton Density Fat Fraction',
        'short': 'fat_quant',
        'plane': 'ax',
        'contrast': '0'
    },
    '22': {
        'long': 'Water Density Fat Fraction',
        'short': 'water_fat_quant',
        'plane': 'ax',
        'contrast': '0'
    },
    '23': {
        'long': 'Portal Venous T1w',
        'short': 'portal_venous',
        'plane': 'ax',
        'contrast': '1'
    },
    '24': {
        'long': 'Coronal Precontrast Fat Suppressed T1w',
        'short': 't1_fat_sat',
        'plane': 'cor',
        'contrast': '0'
    },
    '25': {
        'long': 'Axial Precontrast Fat Suppressed T1w',
        'short': 't1_fat_sat',
        'plane': 'ax',
        'contrast': '0'
    },
    '26': {
        'long': 'R*2',
        'short': 'r_star_2',
        'plane': 'ax',
        'contrast': '0'
    },
    '27': {
        'long': 'Axial Steady State Free Precession',
        'short': 'ssfse',
        'plane': 'ax',
        'contrast': '0'
    },
    '28': {
        'long': 'Coronal Steady State Free Precession',
        'short': 'ssfse',
        'plane': 'cor',
        'contrast': '1'
    },
    '29': {
        'long': 'Venous Subtraction',
        'short': 'venous_sub',
        'plane': 'ax',
        'contrast': '1'
    },
    '0': {
        'long': 'Axial ADC',
        'short': 'adc',
        'plane': 'ax',
        'contrast': '0'
    },
     '30': {
        'long': 'Axial Post Contrast Fat Suppressed T1w',
        'short': 't1_fat_sat',
        'plane': 'ax',
        'contrast': '1'
    },
    '31': {
        'long': 'Coronal Post Contrast Fat Suppressed T1w',
        'short': 't1_fat_sat',
        'plane': 'cor',
        'contrast': '1'
    },
    '32': {
        'long': 'Post Contrast Fat Suppressed T1w',
        'short': 't1_fat_sat',
        'plane': 'ax/cor',
        'contrast': '1'
      }
  }, 
  # Data cropping and normalization, also converts single channel to 3 channel for the model
    data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        #transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
},


}
