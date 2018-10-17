from model_agnostic.agnostic_model import ModelAgnostic
import keras

class KerasModel(ModelAgnostic):
    def __init__(self, weight_path, load_type, backend="tensorflow"):
        """
        :param str weight_path: Path to the file of weights (i.e weights.h5)
        :param str load_type: Type of loading to take place. This will vary based on how you saved the weights. 
        choose complete if you saved the weights and architecture, otherwise implement create model.
        :param str backend: currently only tensorflow is supported as a backend, however others are coming.
        """
        
        self.backend = backend

        super(KerasModel, self).__init__(None, "keras", weight_path)
        self.load_type = load_type
        if load_type == "complete":
            self.model = keras.models.load_model(weight_path)
        else:
            self.model = self.create_model(weight_path)
            
        if backend == "tensorflow":  
            self.tf = __import__(backend)
            self.graph = self.tf.get_default_graph()

    def create_model(self, weight_path):
        """
        Function which creates the model to load.
        Implement this function if you saved just the model weights and
        not the architecture.
        :param str weight_path: Path to the file of the weights
        """
        pass

    def preprocessing(self, items):
        """ Standard preprocessing function to implement"""
        pass

    def predict(self, formatted_data, batch_size=None):
        """
        :param NumPY array formatted_data: This is typically a numpy array returned from the preprocessing function. 
        :param int batch_size: The number of samples to process at once. None will default to 1. 
        """
        if self.backend == "tensorflow":
            with self.graph.as_default():
                self.result = self.model.predict(formatted_data, batch_size=batch_size)
        else:
            self.result = self.model.predict(formatted_data, batch_size=batch_size)
