# main.py (Final Corrected Version)

import argparse
import logging
from kserve import ModelServer, model_server
from model_handler import ZipMapONNXModel

logging.basicConfig(level=logging.INFO)

# 1. Inherit all of KServe's default CLI flags (e.g., --http_port, --model_name)
parser = argparse.ArgumentParser(parents=[model_server.parser])

# 2. Explicitly add the --model_dir argument, which is required by our __init__ method
#    but is not provided by the default KServe parser.
parser.add_argument(
    "--model_dir", required=True, help="A local path to the model directory."
)

# 3. Parse the arguments. The values will come from the ServingRuntime's `args` field.
args, _ = parser.parse_known_args()

# 4. Instantiate our model handler class. This will now work as `args` contains
#    both `model_name` (from the parent parser) and `model_dir` (from our addition).
model = ZipMapONNXModel(name=args.model_name, model_dir=args.model_dir)

# 5. Start the KServe model server
ModelServer().start([model])