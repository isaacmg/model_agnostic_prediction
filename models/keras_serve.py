import sys 
sys.path.append("..")
from agnostic_model import ModelAgnostic

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
            self.model = self.create_model(weight_path)
        else:
            self.model = self.create_model(weight_path)
        global graph
        self.graph = self.tf.get_default_graph()

    def create_model(self, weight_path):
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