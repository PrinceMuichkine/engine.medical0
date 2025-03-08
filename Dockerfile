FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh

# Add conda to path
ENV PATH=$CONDA_DIR/bin:$PATH

# Set working directory
WORKDIR /app

# Copy environment files first to leverage Docker caching
COPY requirements.txt setup.sh ./

# Create conda environment
RUN conda create -n HealthGPT python=3.10 -y

# Install packages
SHELL ["/bin/bash", "-c"]
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate HealthGPT && \
    pip install -r requirements.txt && \
    pip install flask flask-cors gunicorn

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose the port the API will run on
EXPOSE 5000

# Use conda to run commands
SHELL ["/bin/bash", "-c"]
CMD source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate HealthGPT && \
    gunicorn --bind 0.0.0.0:5000 --timeout 300 --workers 1 api_wrapper:app 