# TODO create simple PyTorch example for unitests 

from model_agnostic.models.pytorch_serve import PytorchModel

class ExamplePredict(PytorchModel):
    def __init__(self, weight_path):
        super().__init__(self, weight_path)
