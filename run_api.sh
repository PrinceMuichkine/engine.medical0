#!/bin/bash
cd /workspace/healthgpt

# Install dependencies if not already installed
pip install flask flask-cors pillow

# Set environment variables for CUDA and PyTorch
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run the API
python api_wrapper.py 