import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

print("Creating a simple scikit-learn model...")

# 1. Create a dummy dataset and train a simple model
X_train = np.random.rand(10, 4).astype(np.float32)
y_train = np.random.randint(0, 3, size=10) # 3 classes
model = RandomForestClassifier(n_estimators=3)
model.fit(X_train, y_train)

# This is important: scikit-learn needs class labels to generate the ZipMap
model.classes_ = np.array(['class_A', 'class_B', 'class_C'])

print(f"Model trained with classes: {model.classes_}")

# 2. Define the input type for the ONNX model
# The model expects a batch of inputs, each with 4 features.
initial_type = [('float_input', FloatTensorType([None, 4]))]

# 3. Convert the model to ONNX
# We explicitly leave 'zipmap': True (which is the default) to ensure it's created.
# options = {id(model): {'zipmap': True}} # Not needed, default is True
onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=13)

# 4. Save the ONNX model to a file
model_filename = 'model/zipmap-model.onnx'
with open(model_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Successfully created '{model_filename}' with a ZipMap operator.")
print("You can verify the ZipMap node using a viewer like Netron.")