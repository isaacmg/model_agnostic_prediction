import unittest
from model_agnostic.example_models.example_pytorch import ExamplePredict
from torch.autograd import Variable 
class TestPyTorchModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
       super(TestPyTorchModel, self).__init__(*args, **kwargs)
       self.yolo3 = ExamplePredict("yolov3.weights")

    def test_pytorch_load(self):
        self.assertTrue(self.yolo3)
    
    def test_preprocessing_method(self):
        self.assertEqual(type(self.yolo3.preprocessing("model_agnostic/example_models/image_example/dock.jpg")), Variable)
        self.assertTrue(self.yolo3.preprocessing("model_agnostic/example_models/image_example/hockey.jpg"))

       
    def test_prediction(self):
        self.assertEqual(1, 1)
       # self.assertEqual(self.resnet.predict()



