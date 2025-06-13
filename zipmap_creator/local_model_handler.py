import kserve
import onnxruntime
import numpy as np
from typing import Dict
import os
import uuid 

# The kserve library parses the V2 request into this object
from kserve import InferRequest

class ZipMapONNXModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.session = None
        self.ready = False
        self.model_dir = None

    def load(self) -> bool:
        model_path = "model" #self.model_dir
        try:
            onnx_files = [f for f in os.listdir(model_path) if f.endswith(".onnx")]
            if not onnx_files:
                raise RuntimeError(f"No .onnx file found in directory {model_path}")
            
            final_model_path = os.path.join(model_path, onnx_files[0])
            print(f"Loading model from: {final_model_path}")
            self.session = onnxruntime.InferenceSession(final_model_path)
            self.ready = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.ready = False
            
        return self.ready

    def predict(self, payload: InferRequest, headers: Dict[str, str] = None) -> Dict:
        if not self.ready:
            raise RuntimeError("Model is not loaded, cannot predict.")

        print(f"Received payload: {payload}")
        infer_input = payload.inputs[0]
        print(f"Input data: {infer_input.data}")
        print (f"Input data shape {infer_input.shape}")

        inputs = np.array(infer_input.data, dtype=np.float32).reshape(-1, 4)
        print(f"Input data reshaped to: {inputs.shape}")
        
        input_name = self.session.get_inputs()[0].name
        print(f"Input name: {input_name}")

        output_names = [o.name for o in self.session.get_outputs()]
        print(f"Output names: {output_names}")

        result = self.session.run(output_names, {input_name: inputs})
        print(f"Model inference result: {result}")
        
        zipmap_output = result[1] 
        print(f"ZipMap output: {zipmap_output}")

        predicted_class_names = [max(prob_dict, key=prob_dict.get) for prob_dict in zipmap_output]
        print(f"Predicted class names: {predicted_class_names}")

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


if __name__ == '__main__':
    local_model_dir = os.path.join(os.path.dirname(__file__), "model")
    model = ZipMapONNXModel("zipmap-model")
    model.model_dir = local_model_dir
    model.load()
    kserve.ModelServer().start([model])