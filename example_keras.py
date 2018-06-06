from agnostic_model import KerasModel

class ChexNet(KerasModel):
    def __init__():
        super(ChexNet, self).__init__(self, "keras", weight_path)

    def create_model(self, model_type):
        """
        Make the model
        """
        pass

    def preprocessing(self, items):
        pass

class SimpleResNet50(KerasModel):
    def __init__():
        super(SimpleResNet50, self).__init__(self, "keras", weight_path)
        from keras.applications.resnet50 import preprocess_input

    def create_model(self):
         model = keras.applications.resnet50.ResNet50
        (include_top=True, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None, classes=1000)
        return model

    def preprocessing(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def process_result(self):
        return self.result 
