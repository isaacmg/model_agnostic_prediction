import unittest
from model_agnostic.models import keras_serve
from model_agnostic.example_models.example_keras import SimpleResNet50
import requests 
import os
import shutil 
import numpy


class TestKerasModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
       super(TestKerasModel, self).__init__(*args, **kwargs)
       self.resnet = SimpleResNet50()


    def test_keras_load(self):
        self.assertTrue(self.resnet.model)
    
    def test_preprocessing_method(self):
        url = 'https://images.pexels.com/photos/356378/pexels-photo-356378.jpeg?cs=srgb&dl=adorable-animal-breed-356378.jpg'
        response = requests.get(url, stream=True)
        with open("example2.jpg", 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        processed = self.resnet.preprocessing("example2.jpg")
        #numpy.save('new.npy', processed)
        self.assertEqual(type(processed), numpy.ndarray)
       
    def test_prediction(self):
       
        self.resnet.predict(numpy.load('new.npy'))
        #numpy.save('teste.npy', self.resnet.result)
        self.assertEqual(type(self.resnet.result), numpy.ndarray)
       # self.assertEqual(self.resnet.predict()

    def test_process_result(self):
        self.resnet.result = numpy.load('teste.npy')
        res = self.resnet.process_result()
        self.assertEqual(res[0](2), "Eskimo_dog")


if __name__ == '__main__':
    unittest.main()
