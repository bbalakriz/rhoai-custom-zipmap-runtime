import onnxruntime as rt
import numpy as np

sess = rt.InferenceSession("model/zipmap_test_model.onnx")
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]

# Prepare a sample input
input_data = np.random.rand(1, 4).astype(np.float32)

# Run inference
results = sess.run(output_names, {input_name: input_data})

print("Inference successful with python onnxruntime:")
print(f"Output Label: {results[0]}")
print(f"Output Probabilities: {results[1]}")