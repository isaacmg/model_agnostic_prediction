import sys 
from caffe2.python import workspace
from model_agnostic.agnostic_model import ModelAgnostic

class Caffe2(ModelAgnostic):
    def __init__(self, init_net_path, predict_net_path):
        with open(init_net_path) as f:
            self.init_net = f.read()
        with open(predict_net_path) as f:
            self.predict_net = f.read()
        self.model = workspace.Predictor(self.init_net, self.predict_net)

    def predict(self, formatted_data):
        self.model.run({'data': formatted_data})
    

        
