import sys 
sys.path.append("..")
from agnostic_model import ModelAgnostic

class Caffe2(ModelAgnostic):
    def __init__(self, weight_path, weight_path2):
        with open("init_net.pb") as f:
            self.init_net = f.read()
        with open("predict_net.pb") as f:
            self.predict_net = f.read()  