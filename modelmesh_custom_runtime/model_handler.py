# model_handler.py

import kserve
import onnxruntime
import numpy as np
from typing import Dict
import os
import uuid
import logging

logging.basicConfig(level=logging.INFO)

class ZipMapONNXModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.session = None
        self.input_name = None
        self.output_names = None
        self.model_dir = None

    def load(self) -> bool:
        # This will be set to /mnt/models by the runtime environment
        # self.model_dir = os.environ.get("MODEL_DIR", "/mnt/models")
        logging.info(f"[{self.name}] Attempting to load ONNX model from directory: {self.model_dir}")
        try:
            onnx_files = [f for f in os.listdir(self.model_dir) if f.endswith(".onnx")]
            if not onnx_files:
                raise FileNotFoundError(f"No .onnx file found in directory {self.model_dir}")
            
            model_path = os.path.join(self.model_dir, onnx_files[0])
            self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            self.ready = True
            logging.info(f"[{self.name}] Model loaded successfully.")
        except Exception as e:
            logging.error(f"[{self.name}] Failed to load model: {e}")
            self.ready = False
        return self.ready

    async def predict(self, request: kserve.InferRequest, headers: dict = None) -> Dict:
        if not self.ready:
            raise kserve.errors.InferenceError("Model is not ready.")

        try:
            infer_input = request.inputs[0]
            byte_values_as_int_list = [int(f) for f in infer_input.data]
            raw_byte_data = bytes(byte_values_as_int_list)
            input_data = np.frombuffer(raw_byte_data, dtype=np.float32)
            inputs = input_data.reshape(infer_input.shape)
            
            result = self.session.run(self.output_names, {self.input_name: inputs})
            
            zipmap_output = result[1] 
            predicted_class_names = [max(prob_dict, key=prob_dict.get) for prob_dict in zipmap_output]

            response_id = request.id or str(uuid.uuid4())
            
            # ====================================================================
            # THE FINAL FIX:
            # The returned dictionary MUST exactly match the gRPC protobuf structure.
            # The data goes inside a nested "contents" dictionary.
            # ====================================================================
            return {
                "id": response_id,
                "model_name": self.name,
                "outputs": [
                    {
                        "name": "prediction",
                        "shape": [len(predicted_class_names)],
                        "datatype": "BYTES",
                        "contents": {
                            "bytes_contents": [label.encode('utf-8') for label in predicted_class_names]
                        }
                    }
                ]
            }
        except Exception as e:
            logging.error(f"CRITICAL ERROR IN PREDICT: {e}")
            logging.error(f"DEBUG: Received 'infer_input.data': {request.inputs[0].data}")
            logging.error(f"DEBUG: Received 'infer_input.shape': {request.inputs[0].shape}")
            raise e