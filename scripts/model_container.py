import pickle
#import torch

from config import model_paths
from cnn.cnn_inference import load_pixel_model

class ModelContainer:
    def __init__(self):
        self.cnn_model = load_pixel_model(model_paths['cnn'])
        self.nlp_model = self.load_model(model_paths['nlp'])
        self.metadata_model = self.load_model(model_paths['meta'])
        self.fusion_model = self.load_fusion_model(model_paths['fusion'])
        # self.part_fusion_model = self.load_partial_fusion_model(model_paths['fusion_no_nlp'])

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    
    def load_fusion_model(self, model_path):
        with open(model_path, 'rb') as file:
            from fusion_model.fus_model import FusionModel  #Moved here to avoid circular
            model = pickle.load(file)
        return model
    ...
    # def load_partial_fusion_model(self, model_path):
    #      part_fusion_model = FusionModel(models, num_classes=19, include_nlp=False)
    #      part_fusion_model.load_state_dict(torch.load(model_path))
    #      return part_fusion_model