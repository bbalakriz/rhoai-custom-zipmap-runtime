# main.py

import argparse
import logging
from kserve import ModelServer, model_server

# We import the custom model handler class we wrote
from model_handler import ZipMapONNXModel

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# 1. Initialize the argument parser, inheriting all the standard KServe arguments
#    like --http_port, --grpc_port, and --model_name.
parser = argparse.ArgumentParser(parents=[model_server.parser])

# 2. Add the --model_dir argument, which our handler requires to know
#    where to find the model files inside the pod.
parser.add_argument(
    "--model_dir", required=True, help="A local path to the model directory."
)

# 3. Parse all known arguments passed from the ServingRuntime YAML
args, _ = parser.parse_known_args()

# 4. Instantiate your model handler class, passing it the parsed arguments
model = ZipMapONNXModel(name=args.model_name)
model.model_dir = args.model_dir # Set the model directory on the instance

# 5. Start the ModelServer. It will automatically listen for gRPC requests
#    on the port specified by the --grpc_port argument.
ModelServer().start([model])