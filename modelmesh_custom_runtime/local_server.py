# local_server.py

from kserve import ModelServer
from model_handler import ZipMapONNXModel
import os

if __name__ == "__main__":
    # Point to the local directory where the .onnx file is located
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")

    # Instantiate your model class
    # The __init__ method will load the model from the specified directory
    model = ZipMapONNXModel(name="zipmap-model")
    model.model_dir = model_dir
    model.load()

    # Start a gRPC server on port 8085, which is what ModelMesh expects
    ModelServer(grpc_port=8085).start([model])
