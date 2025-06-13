# model_handler.py (Updated for main.py entrypoint)

import kserve
import onnxruntime
import numpy as np
from typing import Dict
import os
import uuid
import logging

from kserve import InferRequest

logging.basicConfig(level=logging.INFO)

class ZipMapONNXModel(kserve.Model):
    def __init__(self, name: str, model_dir: str):
        """
        Constructor: Loads the model upon instantiation.
        """
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.ready = False
        
        # --- Model loading logic is now here ---
        try:
            logging.info(f"[{self.name}] Attempting to load ONNX model from directory: {self.model_dir}")
            onnx_files = [f for f in os.listdir(self.model_dir) if f.endswith(".onnx")]
            if not onnx_files:
                raise FileNotFoundError(f"No .onnx file found in directory {self.model_dir}")
            
            model_path = os.path.join(self.model_dir, onnx_files[0])
            logging.info(f"[{self.name}] Found model file: {model_path}")
            
            self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            self.ready = True
            logging.info(f"[{self.name}] Model loaded successfully and is now ready.")
        except Exception as e:
            logging.error(f"[{self.name}] Failed to load model: {e}")
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

        infer_input = payload.inputs[0]
        inputs = np.array(infer_input.data, dtype=np.float32).reshape(-1, 4)
        result = self.session.run(self.output_names, {self.input_name: inputs})
        zipmap_output = result[1] 
        predicted_class_names = [max(prob_dict, key=prob_dict.get) for prob_dict in zipmap_output]
        response_id = payload.id or str(uuid.uuid4())
        
        return {
            "id": response_id,
            "model_name": self.name,
            "outputs": [
                {
                    "name": "prediction",
                    "shape": [len(predicted_class_names)],
                    "datatype": "BYTES",
                    "data": [label.encode('utf-8') for label in predicted_class_names]
                }
            ]
        }