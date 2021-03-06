import sys 
sys.path.append("..")
from agnostic_model import ModelAgnostic
import tensorflow as tf 

class TensorflowModel(ModelAgnostic):
    def __init__(self, weight_path, load_type, backend="tensorflow"):
        self.graph = tf.Graph() 
        self.session = tf.Session()
        if load_type == "partial":
            self.model = self.create_model()

        self.saver = tf.train.Saver()
        
    def create_model(self, weight_path):
        """
        Function which creates the model to load.
        Implement this function if you saved just the model weights and
        not the architecture.
        """  
        #self.model = self.saver.restore(self.session, tf.train.load_checkpoint(weight_path))

    def preprocessing(self, items):
        pass

    def predict(self, formatted_data, batch_size=None):
        feed_dict = {}
        self.session.run(self.model, feed_dict)

    def export_tf_serving(self, export_dir:str):
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(self.sess, [tf.saved_model.tag_constants.TRAINING],
                                         strip_default_attrs=True)
        builder.save()


        