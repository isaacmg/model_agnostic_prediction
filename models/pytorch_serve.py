import sys 
sys.path.append("..")
from agnostic_model import ModelAgnostic

class PytorchModel(ModelAgnostic):
    def __init__(self, weight_path):
        self.torch = __import__('torch')
        super().__init__(weight_path, "PyTorch")
        self.model = self.create_model()
        self.model = self.model.load_state_dict('weight')

    def create_model(self):
        pass 

    def predict(self, formatted_data):
        self.model(formatted_data)
    
    def preprocessing(self, items):
        pass
