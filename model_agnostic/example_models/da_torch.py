# TODO create simple PyTorch example for unitests 

from model_agnostic.models.pytorch_serve import PytorchModel
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import numpy as np
class DAPredict(PytorchModel):
    def __init__(self, weight_path):
        super().__init__(self, weight_path)

    def preprocessing(self, data: pd.DataFrame):
        feats = data.as_matrix()
        out_size = data.shape[1]
        
        return feats
            
        
