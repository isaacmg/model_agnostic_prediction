
from model_agnostic.models.keras_serve import KerasModel
from model_agnostic.example_models.chexnet_files.model_factory import ModelFactory
from model_agnostic.example_models.chexnet_files.model_factory import get_model
import numpy as np
import pandas as pd
from keras.utils import Sequence
from PIL import Image
from skimage.transform import resize
import keras

class ChexNet(KerasModel):
    def __init__(self, weight_path):
        self.class_names =  ["Atelectasis","Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nod", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema","Fibrosis", "Pleural_Thickening" ,"Hernia"]
        super(ChexNet, self).__init__(weight_path, "create")
        

    def create_model(self, weight_path):
        """
        Function to Make the model
        """
        
        model_factory = ModelFactory()
        model = get_model(
        the_model = model_factory.models_,
        class_names=self.class_names,
        model_name= "DenseNet121",
        use_base_weights=False,
        weights_path=weight_path)
        return model 

    def preprocessing(self, image_path):
        image = Image.open(image_path)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, (224,224))
        #preprocess_input = self.keras.applications.resnet50.preprocess_input
        
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    
    def process_result(self):
        index = 0 
        for i in self.result[0][0][0]:
            print(i)
            print(self.class_names[index])
            index +=1  
        

class SimpleResNet50(KerasModel):
    def __init__(self, weight_path=None):
        super(SimpleResNet50, self).__init__(weight_path, "create")

    def create_model(self, weight_path):
        model = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        return model

    def preprocessing(self, image_path):
        image = getattr(__import__('keras.preprocessing', fromlist=['image']), 'image')
        preprocess_input = keras.applications.resnet50.preprocess_input
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
        decode_predictions = keras.applications.resnet50.decode_predictions
        return decode_predictions(self.result, top=3)[0]

# TODO MOVE TO TEST FILE
#d = ChexNet("chexnet_files/brucechou1983_CheXNet_Keras_0.3.0_weights.h5")
#result = d.preprocessing("image_example/dock.jpg")
#d.predict(result)
#d.process_result()

