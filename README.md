# Stanford CS231N_project


## Background:
Typical abdominal MRI examinations are comprised of numerous imaging series, each containing several images and each with its own set of features that determines tissue contrast. These differences are largely due to differences in acquisition parameters; additionally, several imaging series performed after contrast administration have timing such that the series can be generally characterized as being from a particular phase of post contrast imaging (e.g., ‘arterial’, ‘portal venous’, ‘hepatobiliary phase’, or ‘delayed’). Accurate automated series identification of abdominal MRI series is important for a number of applications, including display protocols (or, “hanging“ protocols) and advanced postprocessing for use with machine learning or radiomics. The use of the textual series description has limitations; in particular, textual series description can be inaccurate or may be different based on scans performed on different machines (same or different vendors) or using different protocols (e.g., liver protocol versus renal protocol), even within a single institution. A pixel-based classifier may be computationally expensive and those reported in the literature have so far failed to show sufficient accuracy. A metadata model would be limited in that many of the pre and post contrast series would be expected to have the same image acquisition parameters and thus be indistinguishable to a metadata classifier. 

Note that with respect to ‘hanging’ protocols in PACS, this is typically performed using rules-based processes using the series description text and/or parameter values (T1, T2 settings) and may have problems when encountering data with variation. When the hanging protocol fails and there are several empty panels, this requires that the Radiologist finds them which is less efficient, and may cause him/her to not identify series of images which could lead to missed or incorrect diagnoses. 


## Approach
I made use of a CNN model to classify abdominal MRI series into one of 19 distinct classes. Specifically, I am using transfer learning using a base ResNet50 or DenseNet121 architecture fine-tuned for this task. To reduce computational overhead, the model is trained on a single image from each series (the 'middle' image halfway between the first and last in a given series), which parallels a recently published similar approach to classify brain MRI series [2].  This could be used a standalone model or with a fusion model, combined with a DICOM metadata based classifier (similar to the one proposed by Gauriau et al. for brain MRI [1]). 

## Dataset:
The dataset is identical to that reported in [3] and is comprised of scans from multiple vendors and field strength scanners at a single institution. It is representative of typical MRI series from clinical abdominal MRI examinations. For each subject there is a single examination, which is typically comprised of 10-15 different series, and in each series there may be a few to more than 100 images of the same type. For series in which more than one set of parameters may be present (such as series containing diffusion weighted imaging with two b-values, or combined dynamic post-contrast series with multiple phases), the subgroups will be separated into distinct series to classify them separately. The original dataset contains 2,215 MRI series for 105 subjects with each subject having a single examination. The dataset was annotated for the series labels by three radiologists with 2-15 years of experience in abdominal imaging, with 28 options for series type.  Nonstandard MRI series used in research protocols and series types with less than 10 examples will be excluded; the training and testing datasets will be randomly selected from the remaining 2165 remaining series with an 80/20 split at the subject-level resulting in 1733 and 432 series, respectively, each with a single label for the series type. 

## Methods and Results:


## Metadata Classifier
The metadata classifier is a RandomForest model. A grid search is used to tune hyperparameters, and the model is trained on the resultant optimized model. This can be quickly trained on a cpu, and has fairly high accruacy for many of the types of images. It does not, however, do well classifying post contrast series (e.g., portal venous phase, arterial, equilibrium) nor the precontrast series (T1 fat sat) that is performed with identical imaging parameters to the post contrast images. 

![img.tif](/assets/FigCM_meta02230406.tif)


## Pixel-based Classifier
Results from current model:
![img.png](/assets/figures/FigPixel20230322.png)

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
cd app
streamlit run demo.py
```
The streamlit demo app will allow one to view images from the sample batch of studies in the default folder in the left sidebar. These images may or may not have labels embedded into the DICOM tags from prior label processing (generally, the prediction will show over the top left aspect of the image if it has been previously processed). One use of the demo app is to select studies to process (one study/patient at a time). This will generate predictions and write them into the DICOM tags by default. If the destination folder selctor is left blank, the default is for the images to be written back to the same folder, overwriting the previously unprocessed study. Other functions in the demo include the ability to get predictions (the fusion model and its subcomponents) for a single image. It is also possible to view a stuby by the series labels (part of the study in the SeriesDescription), or by the predicted class if the study has been previously processed by the classifier. Overall, the goal is to have a pass-through DICOM server that performs the predictions and sends the processed images back to the souce, but this current demo shows proof of concept and provides a user interface to interact with a study of choice. 

**4. Script process_tree.py**

This is what is called by the demo app to process the studies, but could also be called from the command line by
```
cd scripts
python process_tree.py
```
This provides the user more control and allows for processing of an entire directory of studies, and can set behavior like whether previously processed studies should be re-processed (or skipped), and if the desire is to write over previous tags if they are present. 

## Repository Structure
```


```


## References:
1.	Gauriau R, Bridge C, Chen L, Kitamura F, Tenenholtz NA, Kirsch JE, Andriole KP, Michalski MH, Bizzo BC: Using DICOM Metadata for Radiological Image Series Categorization: a Feasibility Study on Large Clinical Brain MRI Datasets,  Journal of Digital Imaging (2020) 33:747-762.
2.	Zhu Z, Mittendorf A, Shropshire E, Allen B, Miller CM, Bashir MR, Mazurowski MA: 3D Pyramid Pooling Network for Liver MRI Series Classification,   IEEE Trans Pattern Anal Mach Intell. 2020 Oct 28. PMID 33112740.
3.	Cluceru J, Lupo JM, Interian Y, Bove R, Crane JC: Improving the Automatic Classification of Brain MRI Acquisition Contrast with Machine Learning, Journal of Digital Imaging, July 2022.
