from abc import ABC

class ModelFactory(object):
    def infer_model_type(self, weight_path):
        # TODO infers the model given the weight extension
        extension_to_model_type = {".h5":"keras", ".pth":"torch", ".pt":"torch", ".pb":"tensorflow"}
        result = extension_to_model_type[self.weight_path.split('.')[1]]
        return result
    def create_model(self, model_type="inference"):
        if model_type == "inference":
            model_type = self.infer_model_type()
        loader = __import__(model_type, fromlist=[''])
        # TODO need to dynamicall load model given the different weights.
        model = loader.save()
        return model
    def manage_models(self):
        pass 

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
    
        
