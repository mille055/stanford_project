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

class FusionModel(nn.Module):
    def __init__(self, model_container, num_classes, include_nlp = True):
        super(FusionModel, self).__init__()
        self.model_container = model_container
        self.include_nlp = include_nlp
        self.num_inputs = num_classes * 3 if self.include_nlp == True else num_classes * 2
        self.fusion_layer = nn.Linear(self.num_inputs, num_classes)

    def forward(self, x1, x2, x3=None):
        if x3 is not None:
            x = torch.cat((x1, x2, x3), dim=0)
        else:
            x = torch.cat((x1, x2), dim=0)

        x = self.fusion_layer(x)
        return x

    def get_fusion_inference(self, row, device, features, num_classes, classes):
        pred1, prob1 = get_meta_inference(row, self.model_container.metadata_model, features) 
        prob1_tensor = torch.tensor(prob1, dtype=torch.float32).squeeze()

        pred2, prob2 = pixel_inference(self.model_container.cnn_model, [row.fname], classes=classes)
        prob2_tensor = torch.tensor(prob2, dtype=torch.float32)

        pred3, prob3 = get_NLP_inference(self.model_container.nlp_model, [row.fname], device, classes=classes)
        prob3_tensor = torch.tensor(prob3, dtype=torch.float32)

        fused_output = self(prob1_tensor, prob2_tensor, prob3_tensor)

        predicted_class = classes[torch.argmax(fused_output, dim=0).item()]
        confidence_score = torch.max(torch.softmax(fused_output, dim=0)).item()

        troubleshoot_df = pd.DataFrame({'meta_preds': pred1, 'meta_probs': prob1, 'pixel_preds':pred2, 'pixel_probs': prob2, 'nlp_preds': pred3, 'nlp_probs': prob3})

        return predicted_class, confidence_score, troubleshoot_df