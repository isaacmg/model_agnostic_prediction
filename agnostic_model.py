
class ModelAgnostic(object):
    def __init__(self, container_url=None, predict_function, init_model, import_type="inference", weight_path):
        self.container_url = container_url
        if not container_url:
            self.weight_path = weight_path
            self.model = init_model(weight_path, import_type)
        self.predict_function = preprocess_function
        self.result = None
        self.import_type = import_type

    def init_model(self, model_type="inference"):
        if model_type is "inference":
            res = infer_model_type(self.weight_path)
            loader = __import__(res, fromlist=[''])
            return
        else:



    def infer_model_type(self):
        # TODO infers the model given the weight extension
        extension_to_model_type = {".h5":"keras", ".pth":"pytorch", ".pt":"torch" ".pb":"tensorflow"}
        result = extension_to_model_type[self.weight_path.split('.')[1]]
        return result
