
# Use Red Hat's Universal Base Image for Python 3.9
FROM registry.redhat.io/ubi8/python-39

USER root

# Add labels for OpenShift to identify the application
LABEL name="rhoai-zipmap-onnx-runtime" \
      version="1.0" \
      summary="Custom KServe runtime for ONNX models with ZipMap output" \
      description="This runtime uses onnxruntime to serve a model and correctly parses ZipMap responses."

# Create a non-root user and group for security
RUN groupadd -g 10001 app-user && \
    useradd -r -u 10001 -g app-user app-user

# Create a directory for the application
WORKDIR /opt/app-root
COPY --chown=1001:0 . .

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Switch to the non-root user
USER 10001

# Expose the port the server will run on (matches ServingRuntime)
EXPOSE 8001