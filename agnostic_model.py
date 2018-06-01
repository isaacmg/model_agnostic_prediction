
class ModelAgnostic(object):
    def __init__(self, container_url=None, predict_function, init_model, weight_path):
        self.container_url = container_url
        if not container_url:
            self.weight_path = weight_path
            self.model = init_model(weight_path)
        self.predict_function = preprocess_function
        self.result = None


    def init_model(self, model_type="infer"):

    def infer_model_type(self):
        # TODO infers the model given the weight extension
        extension_to_model_type = {} 
        self.weight_path.split('.')[1]

        return
