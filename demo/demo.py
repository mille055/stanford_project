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


from demo_utils import check_prediction_tag, load_dicom_data, apply_window_level, normalize_array, get_single_image_inference

import sys
sys.path.append("../scripts/")
from  process_tree import Processor 
from  fusion_model.fus_model import FusionModel # Import your machine learning model function
from fusion_model.fus_inference import  get_fusion_inference_from_file
from  config import *
from utils import *
from  model_container import ModelContainer

#get instances of model for call to process
model_container = ModelContainer()
fusion_model = FusionModel(model_container = model_container, num_classes=19)

# the place to find the image data
start_folder = "/volumes/cm7/start_folder"

# the place to put processed image data
destination_folder = st.sidebar.text_input("Enter destination folder path:", value="")

# instantiate the processor class for action on the DICOM images
#processor = Processor(old_data_site, destination_site, fusion_model=fusion_model, write_labels=True)
#new_processed_df = processor.pipeline_new_studies()

st.set_page_config(page_title="Abdominal MRI Series Classifier", layout="wide")

st.title("Abdominal MRI Series Classifier")
st.subheader("AIPI540 Project, Spring 2023")
st.write("Chad Miller")


# check for dicom images within the subtree and build selectors for patient, exam, series
if os.path.exists(start_folder) and os.path.isdir(start_folder):
    folder = st.sidebar.selectbox("Select a source folder:", os.listdir(start_folder), index=0)
    selected_folder = os.path.join(start_folder, folder)

    #dest_folder = st.sidebar.input("Select a destination folder")

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

            #selected_class = st.selectbox("Find series type within this study", classes)


            st.subheader("Selected Study Images")
            cols = st.columns(4)

            # Move the window level and image scroll controls below the image
            window_center = st.slider("Window Center", min_value=-1024, max_value=1024, value=0, step=1)
            window_width = st.slider("Window Width", min_value=1, max_value=4096, value=4096, step=1)

            
            image_idx = st.select_slider("View an image", options=range(len(selected_images)), value=0)
            
            
                
            # read in the dicom data for the current images and see if there are labels in the DICOM metadata
            image_path = selected_images[image_idx]
            # Convert to string if necessary
            if not isinstance(image_path, str):
                image_path = str(image_path)            

            # Check if it's a valid file path
            if os.path.isfile(image_path):
                print(f"{image_path} is a valid file path.")
            else:
                print(f"{image_path} is not a valid file path.")
            
            dcm_data = pydicom.dcmread(image_path)
            predicted_type, meta_prediction, cnn_prediction, nlp_prediction  = check_prediction_tag(dcm_data)

            # if not predicted_type: 
            #     print('selected image', image_path)
            #     single_image_df = pd.DataFrame.from_dicoms([image_path])
            #     print('printing what is sent to fus_inference', [str(single_image_df.fname)])
            #     single_image_df, _ = preprocess(single_image_df, model_container.metadata_scaler)
            #     st.dataframe(single_image_df)
            #     predicted_series_class, predicted_series_confidence, ts_df = fusion_model.get_fusion_inference(single_image_df)
            #     predicted_type = abd_label_dict[str(predicted_series_class)]
            #     prediction_meta = abd_label_dict[str(ts_df['meta_preds'])]
            #     cnn_prediction = abd_label_dict[str(ts_df['pixel_preds'])]
            #     nlp_prediction = abd_label_dict[str(ts_df['nlp_preds'])]


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
                        
                        if meta_prediction:
                            textm = f'Metadata prediction: {meta_prediction}'
                            draw.text((15, 50), textm, fill="white")
                        if cnn_prediction:
                            textc = f'Pixel-based CNN prediction: {cnn_prediction}'
                            draw.text((15, 100), textc, fill="white")
                        if nlp_prediction:
                            textn = f'Text-based NLP prediction: {nlp_prediction}'
                            draw.text((15, 150), textn, fill="white")
                    else:
                        draw = ImageDraw.Draw(image)
                        text = f'No prediction yet'
                        draw.text((10,10), text, fill='white')
                    st.image(image, caption=os.path.basename(image_file), use_column_width = True)
                    
                except Exception as e:
                    pass
            
            
            # # If the image is already processed, show the predicted type above the image
            # if predicted_type:
            #     st.write(f"Predicted Type: {predicted_type}")

            # # If the image is not processed, show the button to process the examination in the sidebar
            # else:
            #     predicted_type = 'not implemented yet'
            #     # Display the predicted label
            #     st.write(f"Predicted Type: {predicted_type}")

            #     process_images = st.sidebar.button("Process Images")
            #     if process_images:
            #         processor = Processor(selected_folder, selected_folder, fusion_model=fusion_model, overwrite = True, write_labels=True)
            #         new_processed_df = processor.pipeline_new_studies()
                
            # Now going to show the button all the time, rather than conditionally
            





            process_images = st.sidebar.button("Process Images")
            if process_images:
                if not destination_folder:
                    destination_folder = start_folder
                processor = Processor(selected_folder, destination_folder, fusion_model=fusion_model, overwrite=True, write_labels=True)

                new_processed_df = processor.pipeline_new_studies()
          
            get_inference = st.button("Get Inference For This Image")
            if get_inference:
                # st.write(image_path)
                predicted_type, prediction_meta, cnn_prediction, nlp_prediction = get_single_image_inference(image_path, model_container, fusion_model)
                st.write('Predicted type: ', predicted_type)
                st.write('Metatdata prediction:', prediction_meta)
                st.write('Pixel CNN prediction:', cnn_prediction)
                st.write('Text-based prediction:', nlp_prediction)
        else:
            st.warning("No DICOM files found in the folder.")
else:
    st.error("Invalid start folder path.")


