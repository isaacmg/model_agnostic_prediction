from abc import ABC
class ModelFactory(object):
    def infer_model_type(self):
        # TODO infers the model given the weight extension
        extension_to_model_type = {".h5":"keras", ".pth":"torch", ".pt":"torch", ".pb":"tensorflow"}
        result = extension_to_model_type[self.weight_path.split('.')[1]]
        return result
    def init_model(self, model_type="inference"):
        if model_type is "inference":
            model_type = self.infer_model_type()
        loader = __import__(model_type, fromlist=[''])
        # TODO need to dynamicall load model given the different weights.
        model = loader.save()
        return model

class ModelAgnostic(ABC):
    def __init__(self,  weight_path,  container_url=None, import_type="inference"):
        self.container_url = container_url
        if not container_url:
            self.weight_path = weight_path
            self.model = self.create_model()
        self.result = None
        self.import_type = import_type

    def create_model(self): 
        pass 

    def preprocessing(self, items):
        """
        Must implement this class for your preprocessing needs.
        """
        pass

    def predict(self, formatted_data):
        pass

    def process_result(self, result):
        pass


class PytorchModel(ModelAgnostic):
    def __init__(self, weight_path):
        self.torch = __import__('torch')
        super().__init__(weight_path, "PyTorch")
        self.model = self.create_model()
        self.model = self.model.load_state_dict('weight')

    def create_model(self):
        pass 

    def predict(self, formatted_data):
        self.model(formatted_data)
    
    def preprocessing(self, items):
        pass

class Caffe2(ModelAgnostic):
    def __init__(self, weight_path, weight_path2):
        with open("init_net.pb") as f:
            self.init_net = f.read()
        with open("predict_net.pb") as f:
            self.predict_net = f.read()  

class SciKit(ModelAgnostic):
    def __init__(self):
        pass 

    
        