# RHOAI custom zipmap runtime

## Overview
This project demonstrates how to create a scikit-learn model with a ZipMap output, convert it to ONNX, and serve it using a custom KServe runtime. The workflow includes model creation, testing, and deployment using a custom Docker image and KServe ServingRuntime.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [1. Creating a ZipMap Model](#1-creating-a-zipmap-model)
- [2. Testing the ZipMap Model](#2-testing-the-zipmap-model)
- [3. Building a Custom KServe Runtime](#3-building-a-custom-kserve-runtime)
  - [3.1. Model Handler: Handling ZipMap Inputs/Outputs](#31-model-handler-handling-zipmap-inputsoutputs)
  - [3.2. Creating the Dockerfile](#32-creating-the-dockerfile)
  - [3.3. Deploying with ServingRuntime](#33-deploying-with-servingruntime)
- [References](#references)

---

## Project Structure
```
create_zipmap_test.py
README.md
requirements.txt
kserve_custom_runtime/
    custom_servingruntime_rawdeployment.yaml
    Dockerfile
    inferenceservice_rawdeployment.yaml
    main.py
    model_handler.py
    requirements.txt
model/
    zipmap-model.onnx
modelmesh_custom_runtime/
    Dockerfile
    main.py
    model_handler.py
    requirements.txt
zipmap_creator/
    create_zipmap_model.py
    local_model_handler.py
    test_zipmap_model.py
```

---

## Prerequisites
- Python 3.8+
- Docker
- (Optional) Kubernetes cluster with KServe installed for deployment

Install Python dependencies:
```bash
pip install -r requirements.txt
```

---

## 1. Creating a ZipMap Model

Navigate to the `zipmap_creator` directory and run the model creation script:

```bash
cd zipmap_creator
python create_zipmap_model.py
```

This script will:
- Train a scikit-learn classifier
- Convert it to ONNX format with a ZipMap node (mapping output probabilities to class labels)
- Save the ONNX model as `model/zipmap-model.onnx`

---

## 2. Testing the ZipMap Model

To verify the ONNX model and its ZipMap output, run the test script:

```bash
python test_zipmap_model.py
```

This script will:
- Load the ONNX model
- Run inference on test data
- Print the output, showing the mapping from class labels to probabilities

---

## 3. Building a Custom KServe Runtime

### 3.1. Model Handler: Handling ZipMap Inputs/Outputs

The custom runtime is implemented in `kserve_custom_runtime/model_handler.py`. The `predict` method must:
- Accept input data in the expected ONNX format (matching the model's input signature)
- Run inference using ONNX Runtime
- Parse the output dictionary from the ONNX model, which (due to ZipMap) will map class labels (e.g., 'class_A', 'class_B', 'class_C') to their predicted probabilities
- Return a response in the format expected by KServe (usually a JSON with class probabilities or predicted class)

**Example (pseudo-code):**
```python
# ...existing code...
def predict(self, inputs):
    # Preprocess inputs as needed
    # Run ONNX inference
    result = self.session.run(None, {input_name: input_array})
    # result[0] is a list of dicts: [{class_label: probability, ...}]
    class_probs = result[0][0]  # e.g., {'class_A': 0.7, 'class_B': 0.2, 'class_C': 0.1}
    # Format output for KServe
    return {"predictions": class_probs}
# ...existing code...
```

### 3.2. Creating the Dockerfile

The Dockerfile in `kserve_custom_runtime/Dockerfile` should:
- Use a suitable Python base image
- Copy the model handler and requirements
- Install dependencies (including onnxruntime, scikit-learn, skl2onnx, etc.)
- Set the entrypoint to your serving application (e.g., `main.py`)

**Example Dockerfile:**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "main.py"]
```

### 3.3. Deploying with ServingRuntime

- Use the provided `custom_servingruntime_rawdeployment.yaml` to define your custom ServingRuntime in KServe.
- Deploy your model using `inferenceservice_rawdeployment.yaml`.
- Ensure the ONNX model is accessible to the runtime (e.g., mounted as a volume or included in the image).

**Deployment Steps:**
1. Build and push your Docker image to a registry:
   ```bash
   docker build -t <your-repo>/kserve-custom-runtime:latest .
   docker push <your-repo>/kserve-custom-runtime:latest
   ```
2. Update the YAML files to reference your image and model location.
3. Apply the YAMLs to your Kubernetes cluster:
   ```bash
   kubectl apply -f custom_servingruntime_rawdeployment.yaml
   kubectl apply -f inferenceservice_rawdeployment.yaml
   ```

---

## References
- [scikit-learn](https://scikit-learn.org/)
- [skl2onnx](https://github.com/onnx/sklearn-onnx)
- [ONNX Runtime](https://onnxruntime.ai/)
- [KServe Documentation](https://kserve.github.io/)
- [Netron (ONNX Model Viewer)](https://netron.app/)

---

For any issues or contributions, please open an issue or pull request in this repository.
