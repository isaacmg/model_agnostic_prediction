import sys 
sys.path.append("..")
from models.pytorch_serve import PytorchModel

class ChexNetPyTorch(PytorchModel):
    def __init__(self, weight_path):
        super().__init__(self, weight_path)
    
    def preprocessing():
        