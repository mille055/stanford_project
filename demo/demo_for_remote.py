import pydicom
import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk
import glob
import numpy as np
from PIL import Image, ImageDraw
from glob import glob

import sys
sys.path.append("../scripts/")
from  process_tree import Processor 
from  fusion_model.fus_model import FusionModel # Import your machine learning model function
from  config import *
from  model_container import ModelContainer

from azure.storage.blob import BlobServiceClient

connection_string = "your_connection_string"
container_name = "your_container_name"
local_file_path = "path/to/your/local/file"

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

with open(local_file_path, "rb") as data:
    container_client.upload_blob(data=data, name="your_blob_name")


#get instances of model for call to process
model_container = ModelContainer()
fusion_model = FusionModel(model_container = model_container, num_classes=19)

# instantiate the processor class for action on the DICOM images
#processor = Processor(old_data_site, destination_site, fusion_model=fusion_model, write_labels=True)
#new_processed_df = processor.pipeline_new_studies()

st.set_page_config(page_title="Abdominal MRI Series Classifier", layout="wide")

st.title("Abdominal MRI Series Classifier")
st.subheader("AIPI540 Project, Spring 2023")
st.write("Chad Miller")


# Function to check if the image has been processed and return the value in the DICOM tag (0010, 1010)
def get_predicted_type(dcm_data):
    if (0x0011, 0x1010) in dcm_data:
        return abd_label_dict[str(dcm_data[0x0011, 0x1010].value)]['short']  # this gets the numeric label written into the DICOM and converts to text description
    else:
        return None

@st.cache_resource
def load_dicom_data(folder):
    # function for getting the dicom images within the subfolders of the selected root folder
    # assumes the common file structure of folder/patient/exam/series/images
    data = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".dcm"):
                try:
                    dcm_file_path = os.path.join(root, file)
                    dcm_data = pydicom.dcmread(dcm_file_path)
                    data.append(
                        {
                            "patient": dcm_data.PatientName,
                            "exam": dcm_data.StudyDescription,
                            "series": dcm_data.SeriesDescription,
                            "file_path": dcm_file_path,
                        }
                    )
                except Exception as e:
                    st.warning(f"Error reading DICOM file {file}: {e}")

    return pd.DataFrame(data)

# the place to find the image data
start_folder = "/volumes/cm7/start_folder"

# check for dicom images within the subtree and build selectors for patient, exam, series
if os.path.exists(start_folder) and os.path.isdir(start_folder):
    folder = st.sidebar.selectbox("Select a source folder:", os.listdir(start_folder), index=0)
    selected_folder = os.path.join(start_folder, folder)

    # if there are dicom images somewhere in the tree
    if os.path.exists(selected_folder) and os.path.isdir(selected_folder):
        dicom_df = load_dicom_data(selected_folder)

        if not dicom_df.empty:
            # Select patient
            unique_patients = dicom_df["patient"].drop_duplicates().tolist()
            selected_patient = st.selectbox("Select a patient:", unique_patients)

            # Select exam for the selected patient
            unique_exams = dicom_df[dicom_df["patient"] == selected_patient]["exam"].drop_duplicates().tolist()
            selected_exam = st.selectbox("Select an exam:", unique_exams)

            # Select series for the selected exam
            unique_series = dicom_df[
                (dicom_df["patient"] == selected_patient) & (dicom_df["exam"] == selected_exam)
            ]["series"].drop_duplicates().tolist()
            selected_series = st.selectbox("Select a series:", unique_series)

            # Display images for the selected series
            selected_images = dicom_df[
                (dicom_df["patient"] == selected_patient) &
                (dicom_df["exam"] == selected_exam) &
                (dicom_df["series"] == selected_series)
            ]["file_path"].tolist()
            
            # Sort images within each series by filename
            selected_images.sort(key=lambda x: os.path.basename(x))


            st.subheader("Selected Study Images")
            cols = st.columns(4)

            # Move the window level and image scroll controls below the image
            window_center = st.slider("Window Center", min_value=-1024, max_value=1024, value=0, step=1)
            window_width = st.slider("Window Width", min_value=1, max_value=4096, value=4096, step=1)

            
            image_idx = st.select_slider("View an image", options=range(len(selected_images)), value=0)
            
            # for adjusting the W/L of the displayed image
            def apply_window_level(image, window_center, window_width):
                min_value = window_center - window_width // 2
                max_value = window_center + window_width // 2
                image = np.clip(image, min_value, max_value)
                return image

            def normalize_array(arr):
                arr_min, arr_max = arr.min(), arr.max()
                if arr_max != arr_min:
                    return (arr - arr_min) * 255 / (arr_max - arr_min)
                else:
                    return 0
                
            # read in the dicom data for the current images and see if there are labels in the DICOM metadata
            dcm_data = pydicom.dcmread(selected_images[image_idx])
            predicted_type = get_predicted_type(dcm_data)

            with st.container():
            
                image_file = selected_images[image_idx]
                try:
                    dcm_data = pydicom.dcmread(image_file)
                    image = dcm_data.pixel_array
                    image = apply_window_level(image, window_center, window_width)
                    image = Image.fromarray(normalize_array(image))  # Scale the values to 0-255 and convert to uint8
                    #image = Image.fromarray(dcm_data.pixel_array)
                    image = image.convert("L")
                    if predicted_type:
                        draw = ImageDraw.Draw(image)
                        text = f"Predicted Type: {predicted_type}"
                        draw.text((10, 10), text, fill="white")  # You can adjust the position (10, 10) as needed
                    else:
                        draw = ImageDraw.Draw(image)
                        text = f'No prediction yet'
                        draw.text((10,10), text, fill='white')
                    st.image(image, caption=os.path.basename(image_file), use_column_width = True)
                    
                except Exception as e:
                    pass
            
            
            # If the image is already processed, show the predicted type above the image
            if predicted_type:
                st.write(f"Predicted Type: {predicted_type}")

            # If the image is not processed, show the button to process the examination in the sidebar
            else:
                predicted_type = 'not implemented yet'
                # Display the predicted label
                st.write(f"Predicted Type: {predicted_type}")

                process_images = st.sidebar.button("Process Images")
                if process_images:
                    processor = Processor(selected_folder, selected_folder, fusion_model=fusion_model, overwrite = True, write_labels=True)
                    new_processed_df = processor.pipeline_new_studies()
                
                   
        else:
            st.warning("No DICOM files found in the folder.")
else:
    st.error("Invalid start folder path.")



 