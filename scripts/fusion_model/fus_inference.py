import numpy as np
import pandas as pd
import os
import torch


from fusion_model.fus_model import FusionModel, PartialFusionModel
from cnn.cnn_inference import pixel_inference
from metadata.meta_inference import get_meta_inference
from NLP.NLP_inference import get_NLP_inference
from config import feats_to_keep, classes


def get_fusion_inference(row, model1, model2, model3, fusion_model, device,features=feats_to_keep, num_classes=len(classes)):
    #meta_data, pixel_data, text_data = extract_data_for_models(row)
    
    pred1, prob1 = get_meta_inference(row, model1, features) 
    prob1_tensor = torch.tensor(prob1, dtype=torch.float32).squeeze()
    #print(f'shape of prob1_tensor is {prob1_tensor.shape}')
    
    pred2, prob2 = pixel_inference(model2, [row.fname], classes=classes)
    prob2_tensor = torch.tensor(prob2, dtype=torch.float32)
    #print(f'shape of prob2_tensor is {prob2_tensor.shape}')
    
    
    pred3, prob3 = get_NLP_inference(model3, [row.fname], device, classes=classes)
    prob3_tensor = torch.tensor(prob3, dtype=torch.float32)
    #print(f'shape of prob3_tensor is {prob3_tensor.shape}')
    
    # Pass the tensors through the FusionModel
    fused_output=  fusion_model(prob1_tensor, prob2_tensor, prob3_tensor)
    
    # Get the predicted class and confidence score
    predicted_class = classes[torch.argmax(fused_output, dim=0).item()]
    confidence_score = torch.max(torch.softmax(fused_output, dim=0)).item()
    
    return predicted_class, confidence_score