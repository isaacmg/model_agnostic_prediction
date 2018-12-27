# A simple PyTorch modeL for unittest []
# This model is taken from https://github.com/eriklindernoren/PyTorch-YOLOv3 original 
# implementation by Erik Linder-Norén, article by Redmon, Joseph and Farhadi, Ali
 

from model_agnostic.models.pytorch_serve import PytorchModel
from model_agnostic.example_models.darknet.models import Darknet
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torch 
from skimage.transform import resize

class ExamplePredict(PytorchModel):
    def __init__(self, weight_path):
        #Don't use super as it has its own custom loader super().__init__(self, weight_path)
        self.model = self.create_model()
        self.model.load_weights(weight_path)

    def create_model(self):
        model = Darknet("model_agnostic/example_models/darknet/yolov3.cfg", img_size=416)
        cuda = torch.cuda.is_available()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        return model 
    
    def preprocessing(self, image_path):
        
        img = np.array(Image.open(image_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (416,416, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        input_img = Variable(input_img.type(self.Tensor))
        return input_img.unsqueeze(0)
    
    def process_result(self):
        pass

#e = ExamplePredict("model_agnostic/example_models/darknet/yolov3.weights")
#data = e.preprocessing("model_agnostic/example_models/image_example/dock.jpg")
#e.predict(data)



    

        
        