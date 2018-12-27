import unittest
from model_agnostic.example_models.example_pytorch import ExamplePredict

class TestPyTorchModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
       super(TestPyTorchModel, self).__init__(*args, **kwargs)
       self.yolo3 = ExamplePredict("yolov3.weights")

    def test_pytorch_load(self):
        self.assertTrue(self.yolo3)
    
    def test_preprocessing_method(self):
        # TODO implement 
        self.assertEqual(1, 1)
       
    def test_prediction(self):
        self.assertEqual(1, 1)
       # self.assertEqual(self.resnet.predict()



