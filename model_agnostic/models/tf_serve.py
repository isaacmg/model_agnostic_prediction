import sys 
sys.path.append("..")
from agnostic_model import ModelAgnostic
import tensorflow as tf 

class TensorflowModel(ModelAgnostic):
    def __init__(self, weight_path, load_type, backend="tensorflow"):
        # TODO IMPLEMENT
        self.tf = "s"
        
    def create_model(self, weight_path):
        """
        Function which creates the model to load.
        Implement this function if you saved just the model weights and
        not the architecture.
        """
        pass    

    def preprocessing(self, items):
        pass

    def predict(self, formatted_data, batch_size=None):
        self.model.run()