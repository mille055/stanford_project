# AIPI540_individual_project


## Background:
Typical abdominal MRI examinations are comprised of numerous imaging series, each with its own set of features that determines tissue contrast. These differences are largely due to differences in acquisition parameters; additionally, several imaging series performed after contrast administration have timing such that the series can be generally characterized as being from a particular phase of post contrast imaging (e.g., ‘arterial’, ‘portal venous’, ‘hepatobiliary phase’, or ‘delayed’). Accurate automated series identification of abdominal MRI series is important for a number of applications, including display protocols (or, “hanging“ protocols) and advanced postprocessing for use with machine learning or radiomics. The use of the textual series description has limitations; in particular, textual series description can be inaccurate or may be different based on scans performed on different machines (same or different vendors) or using different protocols (e.g., liver protocol versus renal protocol), even within a single institution. A pixel-based classifier may be computationally expensive and those reported in the literature have so far failed to show sufficient accuracy. A metadata model would be limited in that many of the pre and post contrast series would be expected to have the same image acquisition parameters and thus be indistinguishable to a metadata classifier. 

Note that with respect to ‘hanging’ protocols in PACS, this is typically performed using rules-based processes using the series description text and/or parameter values (T1, T2 settings) and may have problems when encountering data with variation (Figure 1 below). When the hanging protocol fails as in the images below, this causes the Radiologist to be less efficient, and may cause him/her to not review a series of images which could lead to missed or incorrect diagnoses. 


## Approach
I made use of a fusion model to classify abdominal MRI series into one of 19 distinct classes. The fusion model was comprised of submodels of a DICOM metadata based classifier (similar to the one proposed by Gauriau et al. for brain MRI [1]), a pixel-based classifier, and a natural language processing model using the text description. This parallels a recently published similar approach to classify brain MRI series [2], but will be employed in abdominal MRI rather than brain MRI and adds the component of the NLP model. 

## Dataset:
The dataset is identical to that reported in [3] and is comprised of scans from multiple vendors and field strength scanners at a single institution. It is representative of typical MRI series from clinical abdominal MRI examinations. For each subject there is a single examination, which is typically comprised of 10-15 different series, and in each series there may be a few to more than 100 images of the same type. For series in which more than one set of parameters may be present (such as series containing diffusion weighted imaging with two b-values, or combined dynamic post-contrast series with multiple phases), the subgroups will be separated into distinct series to classify them separately. The original dataset contains 2,215 MRI series for 105 subjects with each subject having a single examination. The dataset was annotated for the series labels by three radiologists with 2-15 years of experience in abdominal imaging, with 28 options for series type.  Nonstandard MRI series used in research protocols and series types with less than 10 examples will be excluded; the training and testing datasets will be randomly selected from the remaining 2165 remaining series with an 80/20 split at the subject-level resulting in 1733 and 432 series, respectively, each with a single label for the series type. 

## Methods and Results:
The metadata preprocessing and series selection algorithm are recreated from the paper by Gauriau et al. (reference below), in which a Random Forest classifier is trained to predict the sequence type (e.g. T1, T2, FLAIR, ...) of series of images from brain MRI. Such a tool may be used to select the appropriate series of images for input into a machine learning pipeline.
Reference: Gauriau R, et al. Using DICOM Metadata for Radiological Image Series Categorization: a Feasibility Study on Large Clinical Brain MRI Datasets. Journal of Digital Imaging. 2020 Jan; 33:747–762. (link to paper)

## Metadata Classifier

## Pixel-based Classifier
More description to follow....
Results from current model:
![img.png](/assets/figures/FigPixel20230322.png)

## NLP Classifier


## How to install and use the repository code
**1. Clone this repository**
```
git clone https://github.com/mille055/AIPI540_individual_project
```
**2. Install requirements:**
```
pip install -r requirements.txt
```
**3. Change directory to the demo and run the application**
```
streamlit run demo.py
```

## Repository Structure
```
.
├── README.md
├── assets
│   ├── FigCM_meta20230406.tif
│   ├── FigPixel20230412.png
│   └── figures
│       ├── FigPixel20230322.png
│       └── FigPixel20230322.tif
├── data
│   ├── cmm_labels.txt
│   ├── fusion_test.pkl
│   ├── fusion_train.pkl
│   ├── fusion_val.pkl
│   ├── newly_processed_cases.pkl
│   ├── newly_processed_cases042323.pkl
│   ├── newly_processed_cases042423.pkl
│   ├── test_preds012723.csv
│   ├── testfiles.csv
│   ├── trainfiles.csv
│   └── valfiles.csv
├── demo
│   ├── demo.py
│   └── demo_utils.py
├── models
│   ├── fusion_model_weights042223.pth
│   ├── fusion_model_weights042423.pth
│   ├── fusion_model_weights_new.pth
│   ├── fusion_model_weights_no_nlp042223.pth
│   ├── fusion_model_weights_no_nlp042423.pth
│   ├── fusion_model_weights_no_nlp_new.pth
│   ├── meta_04152023.skl
│   ├── metadata_scaler.pkl
│   ├── pixel_model_041623.pth
│   └── text_model20230415.st
├── notebooks
│   ├── AIPI540_IP_fusion_classifier.ipynb
│   ├── AlterDicoms.ipynb
│   ├── Main_abd02_28_2023.ipynb
│   ├── Playing_with_text_labels.ipynb
│   ├── Train_pixel_class.ipynb
│   ├── Train_pixel_class2.ipynb
│   ├── dependencies.ipynb
│   └── trash.ipynb
├── requirements.txt
├── scripts
│   ├── NLP
│   │   ├── NLP_inference.py
│   │   ├── NLP_training.py
│   │   └── __intit__.py
│   ├── __init__.py
│   ├── cnn
│   │   ├── __init__.py
│   │   ├── cnn_data_loaders.py
│   │   ├── cnn_dataset.py
│   │   ├── cnn_inference.py
│   │   ├── cnn_model.py
│   │   └── cnn_training.py
│   ├── config.py
│   ├── fusion_model
│   │   ├── __init__.py
│   │   ├── fus_inference.py
│   │   ├── fus_model.py
│   │   ├── fus_model_old.py
│   │   └── fus_training.py
│   ├── metadata
│   │   ├── __init__.py
│   │   ├── meta_inference.py
│   │   └── meta_training.py
│   ├── model_container.py
│   ├── process_tree.py
│   └── utils.py
└── tree_output.txt

12 directories, 60 files

```


## References:
1.	Gauriau R, Bridge C, Chen L, Kitamura F, Tenenholtz NA, Kirsch JE, Andriole KP, Michalski MH, Bizzo BC: Using DICOM Metadata for Radiological Image Series Categorization: a Feasibility Study on Large Clinical Brain MRI Datasets,  Journal of Digital Imaging (2020) 33:747-762.
2.	Zhu Z, Mittendorf A, Shropshire E, Allen B, Miller CM, Bashir MR, Mazurowski MA: 3D Pyramid Pooling Network for Liver MRI Series Classification,   IEEE Trans Pattern Anal Mach Intell. 2020 Oct 28. PMID 33112740.
3.	Cluceru J, Lupo JM, Interian Y, Bove R, Crane JC: Improving the Automatic Classification of Brain MRI Acquisition Contrast with Machine Learning, Journal of Digital Imaging, July 2022.
