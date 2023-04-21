import pickle

from config import model_paths
from cnn.cnn_inference import load_pixel_model

# set the path for the various models and put into ModelContainer
#cnn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'pixel_model_041623.pth')
#nlp_path = 

# def load_model(model_path):
#     with open(model_path, 'rb') as file:
#         model = pickle.load(file)
#     return model


class ModelContainer:
    def __init__(self):
        self.cnn_model = load_pixel_model(model_paths['cnn'])
        self.nlp_model = self.load_model(model_paths['nlp'])
        self.metadata_model = self.load_model(model_paths['meta'])
        self.fusion_model = self.load_model(model_paths['fusion'])
        self.part_fusion_model = self.load_model(model_paths['fusion_no_nlp'])

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    
