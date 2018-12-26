# A simple PyTorch modeL for unittest []

from model_agnostic.models.pytorch_serve import PytorchModel
from model_agnostic.example_models.darknet.models import Darknet
from torch.autograd import Variable
from PIL import Image
import torch 

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
        input_image = Image.open(image_path)
        input_img = Variable(input_image.type(self.Tensor))
        return input_img

ExamplePredict("model_agnostic/example_models/darknet/yolov3.weights")


    

        
        