from abc import ABC
class ModelFactory(object):
    def infer_model_type(self):
        # TODO infers the model given the weight extension
        extension_to_model_type = {".h5":"keras", ".pth":"torch", ".pt":"torch", ".pb":"tensorflow"}
        result = extension_to_model_type[self.weight_path.split('.')[1]]
        return result
    def init_model(self, model_type="inference"):
        if model_type is "inference":
            model_type = infer_model_type(self.weight_path)
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

class KerasModel(ModelAgnostic):
    def __init__(self, weight_path, load_type):
        self.keras = __import__('keras')
        self.tf = __import__('tensorflow')
        super(KerasModel, self).__init__(None, "keras", weight_path)
        self.load_type = load_type
        global model
        if load_type is "complete":
            self.model = self.keras.models.load_model(weight_path)
        elif load_type is "create":
            self.model = self.create_model()
        else:
            model = self.create_model()
            self.model = model.load_weights(weight_path)
        global graph
        self.graph = self.tf.get_default_graph()

    def create_model(self):
        """
        Function which creates the model to load.
        Implement this function if you saved just the model weights and
        not the architecture.
        """
        pass

    def preprocessing(self, items):
        pass

    def predict(self, formatted_data, batch_size=None):
        with self.graph.as_default():
            self.result = self.model.predict(formatted_data, batch_size=batch_size)

class PytorchModel(ModelAgnostic):
    def __init__(self, weight_path):
        self.torch = __import__('torch')
        super().__init__(weight_path, "PyTorch")
        self.model = self.create_model()

    def create_model(self):
        pass 

    def predict(self, formatted_data):
        self.model(formatted_data)
    
    def preprocessing(self, items):
        pass 
    
        