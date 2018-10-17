import sys 
sys.path.append("..")
from agnostic_model import ModelAgnostic
from graphpipe import remote

class GraphPipeRemote(ModelAgnostic):
    def __init__(self, url):
      self.graph_pipe_url = url
    
    def preprocessing(self, items):
        pass

    def predict(self, formatted_data, batch_size=None):
        y = remote.execute(self.graph_pipe_url, formatted_data)
        self.result = y

    def process_result(self, result_data):
        pass

        