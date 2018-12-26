# A simple PyTorch modeL for unittest []

from model_agnostic.models.pytorch_serve import PytorchModel
from model_agnostic.example_models.darknet.models import Darknet
from torch.autograd import Variable
from PIL import Image
import torch 

class ExamplePredict(PytorchModel):
    def __init__(self, weight_path):
        super().__init__(self, weight_path)

    def create_model(self):
        model = Darknet("config/yolov3.cfg", img_size=416)
        cuda = torch.cuda.is_available()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        return model 
    
    def preprocessing(self, image_path):
        input_image = Image.open(image_path)
        input_img = Variable(input_image.type(self.Tensor))
        return input_img



    

        
        