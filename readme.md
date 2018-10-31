# Agnostic model loader

[![Build Status](https://travis-ci.org/isaacmg/model_agnostic_prediction.svg?branch=master)](https://travis-ci.org/isaacmg/model_agnostic_prediction)

Model Agnostic (MA) provides an easy model instantiation for Flask, Django, and other production Python projects. It also supports exporting models to other common serving platforms. Finally MA, allows you to easily load models and perform predictions without having to worry about the model backends. Moreover, it helps you organize your code for easy updating and maintainance.

## Usage 

To install run 
` pip install model_agnostic `

You can also build from the source by (though we strongly reccomend using pip)

` git clone https://github.com/isaacmg/model_agnostic_prediction.git`

`cd model_agnostic_prediction`

` python setup.py build install ` 

To use create a new class that extends your model's backend subclass. Models are located in the model folder in the "frameworkName_serve" format. 

```python 
from model_agnostic.models.pytorch_serve import PytorchModel
class NewPyTorch(PytorchModel):
    def __init__(self, weight_path=None):
        super(NewPyTorch, self).__init__(weight_path)
        # Handle any model specific loading here 
    def preprocessing(self, raw_data):
        # Apply necessary pre-processing here for your data here.
        pass
    def process_result(self):
        # Implement this function if have any post-processing needs of predictions.
        pass

```
 You will have to implement `preprocessing` and possibly `create_model` and `process_result` depending on your model (see examples for more info).  

## Project Goals
* Allow users to pre-load model weights from any framework with any save configuration

* Works with common frameworks (PyTorch, Keras, Tensorflow, CNTK).

* Same high level functions regardless of backend framework (i.e. preprocess, predict, process_response)

* Provide a standard template which users can extend for their model's specific functionality.

* Easy to use preprocessing functions that user can tweak depending on their model. 

* A model library (of filled out templates pre-setup) so user can deploy standard models easily onto Flask/Django.

* Easy model exporting to ONNX, TF-Serving


# Examples 

* [CheXNet Keras Deployment on Heroku using model agnostic loader](https://github.com/isaacmg/example_keras_heroku)

* [ResNet50 Example](https://github.com/isaacmg/model_agnostic_prediction/blob/143af897897e675b5cfaff60b6d5212963f8cff8/examples2/example_keras.py#L28)

* [S2I PyTorch example](https://github.com/isaacmg/s2i_pytorch_chex)





