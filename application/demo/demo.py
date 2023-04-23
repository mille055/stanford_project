import pydicom
import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk
import glob
import numpy as np
from PIL import Image
from glob import glob

# import scripts
# from  scripts.process_tree import Processor 
# from  scripts.fusion_model.fus_model import FusionModel # Import your machine learning model function
# from  scripts.config import *
# from scripts.model_container import ModelContainer

st.set_page_config(page_title="Abdominal MRI Series Classifier", layout="wide")

st.title("Abdominal MRI Series Classifier")
st.subheader("AIPI540 Project, Spring 2023")
st.write("Chad Miller")


# Function to load DICOM files into a dataframe

# @st.cache(allow_output_mutation=True)
# def load_dicom_data(folder_path):
#     dicom_files = glob(folder_path + "/*/*.dcm")
#     dicom_files.sort()
#     return dicom_files


# Function to check if the image has been processed and return the value in the DICOM tag (0010, 1010)
def get_predicted_type(dcm_data):
    if (0x0011, 0x1010) in dcm_data:
        return dcm_data[0x0011, 0x1010].value
    else:
        return None

@st.cache_resource
def load_dicom_data(folder):
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

start_folder = "/volumes/cm7/start_folder"

if os.path.exists(start_folder) and os.path.isdir(start_folder):
    folder = st.sidebar.selectbox("Select a folder:", os.listdir(start_folder), index=0)
    selected_folder = os.path.join(start_folder, folder)

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
            st.write(selected_images)
            st.write(selected_images.sort())
            # Sort images within each series by filename
            #selected_images.sort(key=lambda x: os.path.basename(x))

            

            st.subheader("Selected Study Images")
            cols = st.columns(4)

            
            image_idx = st.select_slider("View an image", options=range(len(selected_images)), value=0)
            
            dcm_data = pydicom.dcmread(selected_images[image_idx])
            predicted_type = get_predicted_type(dcm_data)

            # If the image is already processed, show the predicted type above the image
            if predicted_type:
                st.write(f"Predicted Type: {predicted_type}")

            # If the image is not processed, show the button to process the examination in the sidebar
            else:
                process_images = st.sidebar.button("Process Images")
                if process_images:
                # Call your model to process the study
                # Replace with the appropriate function call
                # predicted_type = classify_series(selected_images)

                # # Update the DICOM tag (0010, 1010) with the predicted type
                # for image_file in selected_images:
                #     dcm_data = pydicom.dcmread(image_file)
                #     predicted_type = dcm_data[0x0010, 0x1010].value
                #     # dcm_data.save_as(image_file)
                    predicted_type = 'not implemented yet'
                    # Display the predicted label
                    st.write(f"Predicted Type: {predicted_type}")

            def apply_window_level(image, window_center, window_width):
                min_value = window_center - window_width // 2
                max_value = window_center + window_width // 2
                image = np.clip(image, min_value, max_value)

                def normalize_array(arr):
                    arr_min, arr_max = arr.min(), arr.max()
                    return (arr - arr_min) * 255 / (arr_max - arr_min)

                return normalize_array(image)



            with st.container():
            
                image_file = selected_images[image_idx]
                try:
                    dcm_data = pydicom.dcmread(image_file)
                    image = dcm_data.pixel_array
                    image = apply_window_level(image, window_center, window_width)
                    image = Image.fromarray(np.uint8(image))  # Scale the values to 0-255 and convert to uint8
                    st.image(image, caption=os.path.basename(image_file), use_column_width = True)
                    # dcm_data = pydicom.dcmread(image_file)
                    # image = Image.fromarray(dcm_data.pixel_array)
                    # image = apply_window_level(image, window_center, window_width)
                    # image = image.convert("L")
                    # st.image(image, caption=os.path.basename(image_file), use_column_width=True)
                except Exception as e:
                    pass
            
            # Move the window level and image scroll controls below the image
            window_center = st.slider("Window Center", min_value=-1024, max_value=1024, value=0, step=1)
            window_width = st.slider("Window Width", min_value=1, max_value=4096, value=4096, step=1)


            # for idx, image_file in enumerate(selected_images):
            #     with cols[idx % 4]:
            #         dcm_data = pydicom.dcmread(image_file)
            #         image = Image.fromarray(dcm_data.pixel_array)
            #         image = image.convert("L")  # Convert the image to grayscale mode
            #         st.image(image, caption=os.path.basename(image_file), use_column_width=True)
        else:
            st.warning("No DICOM files found in the folder.")
else:
    st.error("Invalid start folder path.")



# 

# def apply_window_level(image, window_center, window_width):
#     min_value = window_center - window_width // 2
#     max_value = window_center + window_width // 2
#     image = np.clip(image, min_value, max_value)

#     def normalize_array(arr):
#         arr_min, arr_max = arr.min(), arr.max()
#         return (arr - arr_min) * 255 / (arr_max - arr_min)

#     return normalize_array(image)

# start_folder = '/volumes/cm7/start_folder'
# dicom_files = load_dicom_images(start_folder)

# patient_list = sorted(list(set([os.path.basename(os.path.dirname(f)) for f in dicom_files])))
# selected_patient = st.sidebar.selectbox("Select a Patient", patient_list)

# exam_list = sorted(list(set([os.path.basename(os.path.dirname(os.path.dirname(f))) for f in dicom_files if selected_patient in f])))
# selected_exam = st.sidebar.selectbox("Select an Exam", exam_list)

# series_list = sorted(list(set([os.path.basename(f) for f in dicom_files if selected_exam in f])))
# selected_series = st.selectbox("Select a Series", series_list)

# selected_images = [f for f in dicom_files if selected_series in f]
# selected_images.sort(key=lambda x: os.path.basename(x))  # Sort images within each series by filename

# # Process images button
# process_images = st.button("Process Images")

# if process_images:
#     # Classify the series with your machine learning model
#     #predicted_type = classify_series(selected_images)  # Replace with the appropriate function call
#     st.write(f"Predicted Type: not implemented yet")

# image_idx = st.slider("Select Image Index", 0, len(selected_images) - 1, 0)

# window_center = st.slider("Window Center", 0, 4096, 2048)
# window_width = st.slider("Window Width", 0, 4096, 4096)

# with st.container():
#     image_file = selected_images[image_idx]
#     try:
#         dcm_data = pydicom.dcmread(image_file)
#         image = dcm_data.pixel_array
#         image = apply_window_level(image, window_center, window_width)
#         image = Image.fromarray(np.uint8(image))
#         st.image(image, caption=os.path.basename(image_file), use_column_width=True)
#     except Exception as e:
#         pass  # Ignore the file and do nothing






# # # Function to load DICOM files into a dataframe
# # def load_dicom_data(folder):
# #     data = []
# #     for root, _, files in os.walk(folder):
# #         for file in files:
# #             if file.lower().endswith(".dcm"):
# #                 try:
# #                     dcm_file_path = os.path.join(root, file)
# #                     dcm_data = dcmread(dcm_file_path)
# #                     data.append(
# #                         {
# #                             "patient": dcm_data.PatientName,
# #                             "exam": dcm_data.StudyDescription,
# #                             "file_path": dcm_file_path,
# #                         }
# #                     )
# #                 except Exception as e:
# #                     st.warning(f"Error reading DICOM file {file}: {e}")

# #     return pd.DataFrame(data)

# # start_folder = "/volumes/cm7/start_folder"

# # if os.path.exists(start_folder) and os.path.isdir(start_folder):
# #     folder = st.sidebar.selectbox("Select a folder:", os.listdir(start_folder), index=0)
# #     selected_folder = os.path.join(start_folder, folder)

# #     if os.path.exists(selected_folder) and os.path.isdir(selected_folder):
# #         dicom_df = load_dicom_data(selected_folder)

# #         if not dicom_df.empty:
# #             st.subheader("Available Studies")
# #             unique_studies = dicom_df[["patient", "exam"]].drop_duplicates()
# #             study_list = [f"{row.patient} - {row.exam}" for _, row in unique_studies.iterrows()]

# #             selected_study = st.selectbox("Select a study:", study_list)

# #             selected_patient, selected_exam = selected_study.split(" - ")

# #             selected_images = dicom_df[
# #                 (dicom_df["patient"] == selected_patient) & (dicom_df["exam"] == selected_exam)
# #             ]["file_path"].tolist()

# #             st.subheader("Selected Study Images")
# #             cols = st.columns(4)

# #             for idx, image_file in enumerate(selected_images):
# #                 with cols[idx % 4]:
# #                     image = Image.open(image_file)
# #                     st.image(image, caption=os.path.basename(image_file), use_column_width=True)
# #         else:
# #             st.warning("No DICOM files found in the folder.")
# # else:
# #     st.error("Invalid start folder path.")



# # # def dir_selector(folder_path='/volumes/cm7/archived/modified/CmmDemoCase6/'):
# # #     while True:
# # #         folder_list = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
# # #         folder_list.insert(0, '..')  # Add a ".." option to go back up one level
# # #         unique_key = 'dir_selector_' + folder_path.replace(os.path.sep, '_')
# # #         selected_folder = st.sidebar.selectbox('Select a folder:', folder_list, index=0, key=unique_key)

# # #         if selected_folder == '..':
# # #             folder_path = os.path.dirname(folder_path)
# # #         else:
# # #             folder_path = os.path.join(folder_path, selected_folder)
# # #             break

# # #     return folder_path
    
# # # def plot_slice(vol, slice_ix):
# # #     fig, ax = plt.subplots()
# # #     plt.axis('off')
# # #     selected_slice = vol[slice_ix, :, :]
# # #     ax.imshow(selected_slice, origin='lower', cmap='gray')
# # #     return fig
    

# # # st.sidebar.title('DieSitCom')
# # # dirname = dir_selector()

# # # if dirname is not None:
# # #     try:
# # #         reader = sitk.ImageSeriesReader()
# # #         dicom_names = reader.GetGDCMSeriesFileNames(dirname)
# # #         reader.SetFileNames(dicom_names)
# # #         reader.LoadPrivateTagsOn()
# # #         reader.MetaDataDictionaryArrayUpdateOn()
# # #         data = reader.Execute()
# # #         img = sitk.GetArrayViewFromImage(data)
    
# # #         n_slices = img.shape[0]
# # #         slice_ix = st.sidebar.slider('Slice', 0, n_slices, int(n_slices/2))
# # #         output = st.sidebar.radio('Output', ['Image', 'Metadata'], index=0)
# # #         if output == 'Image':
# # #             fig = plot_slice(img, slice_ix)
# # #             plot = st.pyplot(fig)
# # #         else:
# # #             metadata = dict()
# # #             for k in reader.GetMetaDataKeys(slice_ix):
# #                 metadata[k] = reader.GetMetaData(slice_ix, k)
# #             df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Value'])
# #             st.dataframe(df)
# #     except RuntimeError:
# #         st.text('This does not look like a DICOM folder!')

