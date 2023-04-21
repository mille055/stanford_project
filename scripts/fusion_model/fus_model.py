import numpy as np
import pandas as pd
import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torchvision
import pydicom


from cnn.cnn_inference import pixel_inference, load_pixel_model
from metadata.meta_inference import get_meta_inference
from NLP.NLP_inference import get_NLP_inference, load_NLP_model
from config import feats_to_keep, classes, model_paths
from model_container import ModelContainer

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
class FusionModel(nn.Module):
    def __init__(self, model_container, pretrained=True, num_classes=len(classes), features=feats_to_keep, classes=classes, include_nlp=True):
        super(FusionModel, self).__init__()
        self.classes = classes
        self.num_classes = num_classes
        self.features = features
        self.include_nlp = include_nlp
        self.model_container = model_container
    
        self.num_inputs = num_classes * 3 if self.include_nlp == True else num_classes * 2

        # Define the layers of the FusionModel
        self.fusion_layer = nn.Linear(self.num_inputs, self.num_classes)

        if pretrained:
            if include_nlp:
                weights_path = self.model_container.fusion_weights_path
            else:
                weights_path = self.model_container.partial_fusion_model_path
            
            self.load_weights(weights_path)

    def load_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path))

    def forward(self, x1, x2, x3=None):
        if x3 is not None:
            x = torch.cat((x1, x2, x3), dim=0)
            
        else:
            x = torch.cat((x1, x2), dim=0)
            #self.fusion_layer.weight = nn.Parameter(self.model_container.fusion_model.fusion_layer.weight)
        x = self.fusion_layer(x)
        return x
    
    ### adding this back into the FusionModel
    def get_fusion_inference(self, row, classes=classes, features=feats_to_keep, device=device, include_nlp=True):
        # get metadata preds,probs
        pred1, prob1 = get_meta_inference(row, self.model_container.metadata_model, features)
        prob1_tensor = torch.tensor(prob1, dtype=torch.float32).squeeze()
        print(pred1)

        # get cnn preds, probs
        pred2, prob2 = pixel_inference(self.model_container.cnn_model, [row.fname], classes=classes)
        prob2_tensor = torch.tensor(prob2, dtype=torch.float32)
        print(pred2)

        # get nlp preds, probs...if statement because thinking about assessing both ways
        if include_nlp:
            pred3, prob3 = get_NLP_inference(self.model_container.nlp_model, [row.fname], device, classes=classes)
            prob3_tensor = torch.tensor(prob3, dtype=torch.float32)
            print(pred3)
            fused_output = self.forward(prob1_tensor, prob2_tensor, prob3_tensor)
        else:
            fused_output = self.forward(prob1_tensor, prob2_tensor)

        predicted_class = classes[torch.argmax(fused_output, dim=0).item()]
        confidence_score = torch.max(torch.softmax(fused_output, dim=0)).item()

        troubleshoot_df = pd.DataFrame({'meta_preds': pred1, 'meta_probs': prob1, 'pixel_preds': pred2, 'pixel_probs': prob2, 'nlp_preds': pred3, 'nlp_probs': prob3, 'SeriesD': row.SeriesDescription})

        return predicted_class, confidence_score, troubleshoot_df
