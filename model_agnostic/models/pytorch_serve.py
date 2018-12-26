import sys 
sys.path.append("..")
from agnostic_model import ModelAgnostic
import torch
from collections import OrderedDict
class PytorchModel(ModelAgnostic):
    def __init__(self, weight_path, load_type=None):
        self.torch = __import__('torch')
        super().__init__(weight_path, "PyTorch")
        if load_type == "full":
            self.model = torch.load(weight_path)
            
        else:
            self.model = self.create_model()

            checkpoint = torch.load(weight_path, map_location= lambda storage, loc: storage)
            new_state_dict = OrderedDict()
            if torch.cuda.device_count() < 1 and "state_dict" not in checkpoint:
                for k, v in checkpoint['state_dict'].items():
                    k = k[7:] # remove `module.`
                
                self.model.load_state_dict(new_state_dict)
            # Here we are assuming that raw state_dict has already been transformed for appropiate loading   
            elif "state_dict" not in checkpoint:
                self.model.load_state_dict(checkpoint)
              
            else:
                self.model = torch.nn.DataParallel(self.model)
                self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def create_model(self):
        pass 

    def predict(self, formatted_data):
        self.result = self.model(formatted_data)
    
    def preprocessing(self, items):
        pass
    
    def to_onnx(self, save_path, formatted_data, precision=3, export_params=True, verify=False):
        torch_out = torch.onnx._export(self,             # model being run
                               formatted_data,                       # model input (or a tuple for multiple inputs)
                               save_path, # where to save the model (can be a file or file-like object)
                               export_params=export_params)
        if verify: 
            self.verify_onnx(torch_out, formatted_data, precision, save_path)

    def verify_onnx(self, torch_out, input_data, precision, model_path):
        import onnx
        import caffe2.python.onnx.backend as onnx_caffe2_backend
        import numpy as np
        model = onnx.load(model_path)
        prepared_backend = onnx_caffe2_backend.prepare(model)
        W = {model.graph.input[0].name: input_data.data.numpy()}
        c2_out = prepared_backend.run(W)[0]
        np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=precision)
        print("Model passed percision test")


            
    