# client.py

import grpc
import numpy as np

# Import the auto-generated gRPC classes from the kserve library
from kserve.protocol.grpc import grpc_predict_v2_pb2
from kserve.protocol.grpc import grpc_predict_v2_pb2_grpc

def main():
    # Define the server address
    server_address = "localhost:8085"
    
    # Create some dummy input data that matches the model's expected shape (1 row, 4 features)
    input_data = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    # 1. Create the gRPC request input tensor
    # This is the most complex part: packing the numpy data into the protobuf message
    infer_input = grpc_predict_v2_pb2.ModelInferRequest.InferInputTensor(
        name="float_input",  # This must match the input name of your ONNX model
        shape=list(input_data.shape),
        datatype="FP32",
        contents=grpc_predict_v2_pb2.InferTensorContents(
            fp32_contents=input_data.tobytes() # Convert numpy array to raw bytes
        )
    )

    # 2. Create the main inference request object
    request = grpc_predict_v2_pb2.ModelInferRequest(
        model_name="zipmap-model",
        id="some-unique-id-123",
        inputs=[infer_input]
    )

    print("--- Sending gRPC Request ---")
    print(request)

    # 3. Create a gRPC channel and a client stub
    channel = grpc.insecure_channel(server_address)
    stub = grpc_predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)

    # 4. Make the call to the server
    response = stub.ModelInfer(request)

    # 5. Parse the response
    print("\n--- Received gRPC Response ---")
    output_tensor = response.outputs[0]
    # The string results are in the 'bytes_contents' field
    predicted_labels = output_tensor.contents.bytes_contents
    
    print(f"Model Name: {response.model_name}")
    print(f"Response ID: {response.id}")
    print(f"Predicted Class: {[label.decode('utf-8') for label in predicted_labels]}")

if __name__ == "__main__":
    main()