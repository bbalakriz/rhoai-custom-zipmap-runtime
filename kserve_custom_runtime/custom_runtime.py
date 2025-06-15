import kserve
from kserve import ModelServer, model_server
import onnxruntime

import numpy as np
from typing import Dict
import os
import uuid
import logging
import argparse


from kserve import InferRequest

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ZipMapONNXModel(kserve.Model):
    def __init__(self, model_name: str, model_directory: str):
        super().__init__(model_name)
        self.name = model_name
        self.model_dir = model_directory
        self.ready = False
        try:
            logger.info(f"[{self.name}] attempting to load ONNX model from: {self.model_dir}")
            onnx_model_files = [f for f in os.listdir(self.model_dir) if f.endswith(".onnx")]
            if not onnx_model_files:
                raise FileNotFoundError(f"No .onnx file found in {self.model_dir}")

            model_path = os.path.join(self.model_dir, onnx_model_files[0])
            
            logger.info(f"[{self.name}] Using model file: {model_path}")    
            self.session = onnxruntime.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.ready = True
            
            logger.info(f"[{self.name}] Model loaded and ready.")
        except Exception as error:
            logger.error(f"[{self.name}] Model loading failed: {error}")
            self.ready = False

    def load(self) -> bool:
        """
        The load method is no longer used, but is required by the interface.
        We return the readiness state set during __init__.
        """
        return self.ready

    def predict(self, payload: InferRequest, headers: Dict[str, str] = None) -> Dict:
        # This V2 predict method remains unchanged
        if not self.ready:
            raise kserve.errors.InferenceError("Model is not ready.")

        # extract the first input from the payload (assumes a single input)
        infer_input = payload.inputs[0]
        
        # convert input data to a NumPy array and reshape for model input
        inputs = np.array(infer_input.data, dtype=np.float32).reshape(-1, 4)
        
        # run inference using the ONNX model session
        result = self.session.run(self.output_names, {self.input_name: inputs})
        
        # second output is the ZipMap output: a list of dicts mapping class labels to probabilities
        zipmap_output = result[1]
        
        # for each prediction, select the class label with the highest probability
        predicted_class_names = [max(prob_dict, key=prob_dict.get) for prob_dict in zipmap_output]
        
        # use the payload ID if present, otherwise generate a new UUID
        response_id = payload.id or str(uuid.uuid4())
        
        # Build and return the response in KServe V2 protocol format
        return {
            "id": response_id,
            "model_name": self.name,
            "outputs": [
                {
                    "name": "prediction",
                    "shape": [len(predicted_class_names)],
                    "datatype": "BYTES",
                    # encode class labels as bytes for the response
                    "data": [label.encode('utf-8') for label in predicted_class_names]
                }
            ]
        }

def main():
    # inherit KServe's default args (--http_port, --model_name , etc.)
    parser = argparse.ArgumentParser(parents=[model_server.parser])

    # add the --model_dir argument
    parser.add_argument(
        "--model_dir", required=True, help="A local path to the model directory."
    )

    args, _ = parser.parse_known_args()
    model = ZipMapONNXModel(model_name=args.model_name, model_directory=args.model_dir)
    ModelServer().start([model])

if __name__ == "__main__":
    main()
