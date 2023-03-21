# Set the base image to the nvidia runtime image for CUDA 11.4 and cuDNN 8
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN echo "deb [trusted=yes] http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN echo "deb [trusted=yes] http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-setuptools \
    git \
    logrotate
    
RUN rm -rf /var/lib/apt/lists/*

# Copy logrotate config file
COPY logrotate.conf /etc/logrotate.d/

# Copy necessary files from local directory to the container
COPY fine_tune_gpu.py requirements.txt /app/

# Set the working directory to /app
WORKDIR /app

# Create a virtual environment and install dependencies from requirements.txt
RUN python3 -m venv venv
RUN . venv/bin/activate && pip install --upgrade pip
RUN . venv/bin/activate && pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
RUN . venv/bin/activate && pip install --no-cache-dir --trusted-host download.pytorch.org torch==1.7.1+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Set the NVIDIA_VISIBLE_DEVICES environment variable to the GPU ID you want to use
ENV NVIDIA_VISIBLE_DEVICES=all

# Run fine_tune_gpu.py and write output and error to log files
CMD . venv/bin/activate && python fine_tune_gpu.py --train_csv $TRAIN_CSV --eval_csv $EVAL_CSV --device $DEVICE > /var/log/output.log 2> /var/log/error.log

# Expose the port for TensorBoard
EXPOSE 6006

# To view tensorboard in http://localhost:6006/
# ssh -L 6006:localhost:6006 user@remote-vm-ip
