# Set the base image to the nvidia runtime image for CUDA 11.4 and cuDNN 8
FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    logrotate \
    && rm -rf /var/lib/apt/lists/*

# Copy logrotate config file
COPY logrotate.conf /etc/logrotate.d/

# Copy necessary files from local directory to the container
COPY fine_tune_gpu.py requirements.txt /app/

# Set the working directory to /app
WORKDIR /app

# Create a virtual environment and install dependencies from requirements.txt
RUN python3 -m venv venv
RUN . venv/bin/activate && pip install -r requirements.txt

# Set the NVIDIA_VISIBLE_DEVICES environment variable to the GPU ID you want to use
ENV NVIDIA_VISIBLE_DEVICES=all

# Run fine_tune_gpu.py and write output and error to log files
CMD . venv/bin/activate && python fine_tune_gpu.py > /var/log/output.log 2> /var/log/error.log

# Expose the port for TensorBoard
EXPOSE 6006

# To view tensorboard in http://localhost:6006/
# ssh -L 6006:localhost:6006 user@remote-vm-ip
