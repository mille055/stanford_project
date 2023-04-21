import numpy as np
import pandas as pd
import os
import torch
import pickle

from fusion_model.fus_model import FusionModel
from cnn.cnn_inference import pixel_inference, load_pixel_model
from metadata.meta_inference import get_meta_inference
from NLP.NLP_inference import get_NLP_inference, load_NLP_model
from config import feats_to_keep, classes, model_paths
from model_container import ModelContainer


# Load the models and create an instance of the ModelContainer
model_container_instance = ModelContainer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def get_fusion_inference(row, models = model_container_instance, device=device,features=feats_to_keep, num_classes=len(classes)):
#     #meta_data, pixel_data, text_data = extract_data_for_models(row)
    
#     pred1, prob1 = get_meta_inference(row, model_container_instance.metadata_model, features) 
#     prob1_tensor = torch.tensor(prob1, dtype=torch.float32).squeeze()
#     #print(f'shape of prob1_tensor is {prob1_tensor.shape}')
    
#     pred2, prob2 = pixel_inference(model_container_instance.cnn_model, [row.fname], classes=classes)
#     prob2_tensor = torch.tensor(prob2, dtype=torch.float32)
#     #print(f'shape of prob2_tensor is {prob2_tensor.shape}')
    
    
#     pred3, prob3 = get_NLP_inference(model_container_instance.nlp_model, [row.fname], device, classes=classes)
#     prob3_tensor = torch.tensor(prob3, dtype=torch.float32)
#     #print(f'shape of prob3_tensor is {prob3_tensor.shape}')
    
#     # choose fusion model and instantiate
#     fusion_model = model_container_instance.fusion_model

#     # Pass the tensors through the FusionModel
#     fused_output=  fusion_model(prob1_tensor, prob2_tensor, prob3_tensor)
    
#     # Get the predicted class and confidence score
#     predicted_class = classes[torch.argmax(fused_output, dim=0).item()]
#     confidence_score = torch.max(torch.softmax(fused_output, dim=0)).item()
    
#     return predicted_class, confidence_score

def get_fusion_inference(row, model_container, classes=classes, features=feats_to_keep, device=device, include_nlp=True):
    # unpack the models
    metadata_model = model_container.metadata_model
    cnn_model = model_container.cnn_model
    nlp_model = model_container.nlp_model
    fusion_model = model_container.fusion_model

    # get metadata preds,probs
    pred1, prob1 = get_meta_inference(row, metadata_model, features)
    prob1_tensor = torch.tensor(prob1, dtype=torch.float32).squeeze()
    print(pred1)

    # get cnn preds, probs
    pred2, prob2 = pixel_inference(cnn_model, [row.fname], classes=classes)
    prob2_tensor = torch.tensor(prob2, dtype=torch.float32)
    print(pred2)

    # get nlp preds, probs...if statement because thinking about assessing both ways
    if include_nlp:
        pred3, prob3 = get_NLP_inference(nlp_model, [row.fname], device, classes=classes)
        prob3_tensor = torch.tensor(prob3, dtype=torch.float32)
        print(pred3)
        fused_output = fusion_model(prob1_tensor, prob2_tensor, prob3_tensor)
    else:
        fused_output = fusion_model(prob1_tensor, prob2_tensor)

    predicted_class = classes[torch.argmax(fused_output, dim=0).item()]
    confidence_score = torch.max(torch.softmax(fused_output, dim=0)).item()

    troubleshoot_df = pd.DataFrame({'meta_preds': pred1, 'meta_probs': prob1, 'pixel_preds': pred2, 'pixel_probs': prob2, 'nlp_preds': pred3, 'nlp_probs': prob3, 'SeriesD': row.SeriesDescription})

    return predicted_class, confidence_score, troubleshoot_df



def load_fusion_model(model_path):
    with open(model_path, 'rb') as file:
        fusion_model = pickle.load(file)
    return fusion_model