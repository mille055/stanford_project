import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk
import glob


def dir_selector(folder_path='/volumes/cm7/archived/modified/CmmDemoCase6/'):
    while True:
        folder_list = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        folder_list.insert(0, '..')  # Add a ".." option to go back up one level
        unique_key = 'dir_selector_' + folder_path.replace(os.path.sep, '_')
        selected_folder = st.sidebar.selectbox('Select a folder:', folder_list, index=0, key=unique_key)

        if selected_folder == '..':
            folder_path = os.path.dirname(folder_path)
        else:
            folder_path = os.path.join(folder_path, selected_folder)
            break

    return folder_path
    
def plot_slice(vol, slice_ix):
    fig, ax = plt.subplots()
    plt.axis('off')
    selected_slice = vol[slice_ix, :, :]
    ax.imshow(selected_slice, origin='lower', cmap='gray')
    return fig
    

st.sidebar.title('DieSitCom')
dirname = dir_selector()

if dirname is not None:
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dirname)
        reader.SetFileNames(dicom_names)
        reader.LoadPrivateTagsOn()
        reader.MetaDataDictionaryArrayUpdateOn()
        data = reader.Execute()
        img = sitk.GetArrayViewFromImage(data)
    
        n_slices = img.shape[0]
        slice_ix = st.sidebar.slider('Slice', 0, n_slices, int(n_slices/2))
        output = st.sidebar.radio('Output', ['Image', 'Metadata'], index=0)
        if output == 'Image':
            fig = plot_slice(img, slice_ix)
            plot = st.pyplot(fig)
        else:
            metadata = dict()
            for k in reader.GetMetaDataKeys(slice_ix):
                metadata[k] = reader.GetMetaData(slice_ix, k)
            df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Value'])
            st.dataframe(df)
    except RuntimeError:
        st.text('This does not look like a DICOM folder!')
