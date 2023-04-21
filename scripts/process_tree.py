import numpy as np
import pandas as pd
import os
from fusion_model.fus_inference import get_fusion_inference
import pydicom
from utils import *
from config import model_paths



# def process_batch(df, data_dir, destination_folder, write_labels=True):

#     df1 = df.copy()
#     #print('In batch, columns are: ', df1.columns)
#     batch = df1.groupby('patientID').apply(lambda x: process_patient(x, data_dir, destination_folder, write_labels))
    
#     # print('writing labels into dicom in location ', dest_name)
#     # for filename in batch.fname:

#     return batch

# def process_patient(patient_df, data_dir, destination_folder, write_labels):
#     processed_exams = patient_df.groupby('exam').apply(lambda x: process_exam(x, data_dir, destination_folder, write_labels))
#     return processed_exams


# def process_exam(exam_df, data_dir, destination_folder, write_labels):
#     # Group exam data by series and apply the process_series function
#     processed_series = exam_df.groupby('series').apply(lambda x: process_series(x, data_dir, destination_folder, write_labels))

#     #result = pd.concat(processed_series)
#     result = processed_series
#     return result
    
# def process_series(series_df, data_dir, destination_folder, write_labels, selection_fraction=0.5):
#     # Sort the dataframe by file_info (or another relevant column)
#     sorted_series = series_df.sort_values(by='fname')

#     # Find the middle image index
#     middle_index = int(len(sorted_series) * selection_fraction)

#     # Get the middle image
#     middle_image = sorted_series.iloc[middle_index]

#     predicted_series_class, predicted_series_confidence = get_fusion_inference(middle_image)

#     sorted_series['predicted_class'] = predicted_series_class
#     sorted_series['prediction_confidence'] = np.round(predicted_series_confidence, 2)

#     #save_path = f'/volumes/cm7/processed/modified/{series_df.patientID}/{series_df.exam}/{series_df["series"]}/'

#      # Define the save path relative to data_dir
#     relative_path = os.path.relpath(series_df.fname.iloc[0], data_dir)
#     save_path = os.path.join(data_dir, destination_folder, os.path.dirname(relative_path))

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     if write_labels:
#         #print('writing new data into', save_path)
#         write_labels_into_dicom(sorted_series, label_num=predicted_series_class,
#                             conf_num=np.round(predicted_series_confidence, 3), path=save_path)

#     return sorted_series
    

class Processor:
    def __init__(self, data_dir, destination_folder, write_labels=True):
        self.data_dir = data_dir
        self.destination_folder = destination_folder
        self.write_labels = write_labels

    def process_batch(self, df):
        df1 = df.copy()
        batch = df1.groupby('patientID').apply(self.process_patient)
        return batch

    def pipeline_new_studies(self):
        _, df = get_dicoms(self.data_dir)
        df1 = df.copy()
        ## manipulate df prior to evaluation
        df1 = expand_filename(df1, ['blank', 'filename', 'series', 'exam', 'patientID'])
        df1.drop(columns='blank', inplace=True)
        df1['file_info']=df1.fname
        df1['img_num'] = df1.file_info.apply(extract_image_number)
        df1['contrast'] = df1.apply(detect_contrast, axis=1)
        df1['plane'] = df1.apply(compute_plane, axis=1)
        df1['series_num'] = df1.series.apply(lambda x: str(x).split('_')[-1])
        #print('columns before preprocess are', df1.columns)
        df1 = preprocess(df1)

        processed_frame = self.process_batch(df1)
        return processed_frame

    def process_patient(self, patient_df):
        processed_exams = patient_df.groupby('exam').apply(self.process_exam)
        return processed_exams

    def process_exam(self, exam_df):
        processed_series = exam_df.groupby('series').apply(self.process_series)
        return processed_series

    def process_series(self, series_df, selection_fraction=0.5):
        # Sort the dataframe by file_info (or another relevant column)
        sorted_series = series_df.sort_values(by='fname')

        # Find the middle image index
        middle_index = int(len(sorted_series) * selection_fraction)

        # Get the middle image
        middle_image = sorted_series.iloc[middle_index]

        predicted_series_class, predicted_series_confidence = get_fusion_inference(middle_image)

        sorted_series['predicted_class'] = predicted_series_class
        sorted_series['prediction_confidence'] = np.round(predicted_series_confidence, 2)

        #save_path = f'/volumes/cm7/processed/modified/{series_df.patientID}/{series_df.exam}/{series_df["series"]}/'

        # Define the save path relative to data_dir
        relative_path = os.path.relpath(series_df.fname.iloc[0], data_dir)
        save_path = os.path.join(data_dir, destination_folder, os.path.dirname(relative_path))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if write_labels:
            #print('writing new data into', save_path)
            write_labels_into_dicom(sorted_series, label_num=predicted_series_class,
                            conf_num=np.round(predicted_series_confidence, 3), path=save_path)

        return sorted_series
    


def write_labels_into_dicom(series_group, label_num, conf_num, path):
    #print('writing labels', label_num)
    for dicom_file in series_group.fname.tolist():
        filename = os.path.basename(dicom_file)
        ds = dcmread(dicom_file, no_pixels=False)

        private_creator_tag = pydicom.tag.Tag(0x0011, 0x0010)
        custom_tag1 = pydicom.tag.Tag(0x0011, 0x1010)
        custom_tag2 = pydicom.tag.Tag(0x0011, 0x1011)

        # Check if private creator and custom tags already exist
        if private_creator_tag not in ds or custom_tag1 not in ds or custom_tag2 not in ds:
            # Create and add private creator element
            private_creator = pydicom.DataElement(private_creator_tag, 'LO', 'PredictedClassInfo')
            ds[private_creator_tag] = private_creator

            # Create and add custom private tags
            data_element1 = pydicom.DataElement(custom_tag1, 'IS', str(label_num))
            data_element1.private_creator = 'PredictedClassInfo'
            data_element2 = pydicom.DataElement(custom_tag2, 'DS', str(conf_num))
            data_element2.private_creator = 'PredictedClassInfo'
            ds[custom_tag1] = data_element1
            ds[custom_tag2] = data_element2

            modified_file_path = os.path.join(path, filename)
            ds.save_as(modified_file_path)
        else:
            print(f"Custom tags already exist in {dicom_file}, skipping this file.")

        modified_file_path = os.path.join(path, filename)
        ds.save_as(modified_file_path)  


# def pipeline_new_image_df(data_dir, dest_name='modified', write_labels = True):
#     # create the df of image data    
#     _, df = get_dicoms(data_dir)
#     df1 = df.copy()
#     ## manipulate df prior to evaluation
#     df1 = expand_filename(df1, ['blank', 'filename', 'series', 'exam', 'patientID'])
#     df1.drop(columns='blank', inplace=True)
#     df1['file_info']=df1.fname
#     df1['img_num'] = df1.file_info.apply(extract_image_number)
#     df1['contrast'] = df1.apply(detect_contrast, axis=1)
#     df1['plane'] = df1.apply(compute_plane, axis=1)
#     df1['series_num'] = df1.series.apply(lambda x: str(x).split('_')[-1])
#     #print('columns before preprocess are', df1.columns)

#     df1 = preprocess(df1)
#    # print('after preprocessin exam is in columns?', ('exam' in df1.columns))
#     #print('after preprocessing series is in columns?', ('series' in df1.columns))
#     #process the batch of studies
#     processor = Processor(data_dir, destination_folder)
#     processed_frame = processor.pipeline_new_image_df()
    
#     processed_frame = process_batch(df1, data_dir, dest_name, write_labels)

#     return processed_frame


def main():
    old_data_site = '/volumes/cm7/processed/'
    destination_site = '/volumes/cm7/newly_processed/'
    processor = Processor(old_data_site, destination_site, write_labels=True)


if __name__ == "__main__":
    main()