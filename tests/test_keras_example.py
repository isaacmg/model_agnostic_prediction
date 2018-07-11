import unittest
from model_agnostic.models import keras_serve
from model_agnostic.example_models.example_keras import SimpleResNet50

class TestKerasModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
       super(TestKerasModel, self).__init__(*args, **kwargs)
       self.resnet = SimpleResNet50()

    def test_keras_load(self):
        self.assertTrue(self.resnet.model)
    
    def test_preprocessing_method(self):
        # TODO implement 
        self.assertEqual(1, 1)
       
    def test_prediction(self):
        self.assertEqual(1, 1)
       # self.assertEqual(self.resnet.predict()



