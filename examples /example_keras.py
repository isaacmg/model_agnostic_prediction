import sys 
sys.path.append("..")
from models.keras_serve import KerasModel
from chexnet_files.model_factory import model_factory

class ChexNet(KerasModel):
    def __init__(self, weight_path):
        super(ChexNet, self).__init__(self, "keras", weight_path)

    def create_model(self, model_type, weight_path):
        """
        Function to Make the model
        """
        class_names = ["Atelectasis","Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nod", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema","Fibrosis", "Pleural_Thickening" ,"Hernia"]
        self.model = model_factory.get_model(
        class_names,
        model_name= "DenseNet121",
        use_base_weights=False,
        weights_path=weight_path)
        pass

    def preprocessing(self, items):
        pass

class SimpleResNet50(KerasModel):
    def __init__(self, weight_path):
        super(SimpleResNet50, self).__init__(weight_path, "create")

    def create_model(self, weight_path):
        model = self.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        return model

    def preprocessing(self, image_path):
        image = getattr(__import__('keras.preprocessing', fromlist=['image']), 'image')
        preprocess_input = self.keras.applications.resnet50.preprocess_input
        import numpy as np 
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def process_result(self):
        """
        Handles the result. In this case returns the top three results.
        """
        decode_predictions = self.keras.applications.resnet50.decode_predictions
        return decode_predictions(self.result, top=3)[0]
