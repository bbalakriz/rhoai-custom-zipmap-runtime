# Dockerfile

# Use a certified Red Hat UBI Python image. We will install KServe ourselves.
FROM registry.redhat.io/ubi8/python-39

USER root
# Create a home directory for the app and set permissions
RUN mkdir -p /opt/app-root && chown -R 1001:0 /opt/app-root
WORKDIR /opt/app-root

# Copy your requirements and install them
COPY --chown=1001:0 requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your model handler code
COPY --chown=1001:0 . .

# Switch to the non-root user
USER 1001

# The ENTRYPOINT from the ServingRuntime will handle the command.