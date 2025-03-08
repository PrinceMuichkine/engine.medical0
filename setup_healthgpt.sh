#!/bin/bash

echo "HealthGPT Setup Script"
echo "======================"
echo ""
echo "This script will help you set up the HealthGPT integration with the Medical0 AI chatbot."
echo ""

# Check for NVIDIA GPU
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi
else
    echo "❌ NVIDIA GPU not detected. HealthGPT requires an NVIDIA GPU with CUDA support."
    echo "You can still continue, but the application might not work properly."
    read -p "Continue anyway? (y/n): " continue_without_gpu
    if [[ "$continue_without_gpu" != "y" ]]; then
        echo "Setup aborted."
        exit 1
    fi
fi

# Create directories for model weights
echo ""
echo "Creating directories for model weights..."
mkdir -p ./clip-vit-large-patch14-336
mkdir -p ./Phi-3-mini-4k-instruct
mkdir -p ./phi-4
mkdir -p ./HealthGPT-M3
mkdir -p ./HealthGPT-L14
mkdir -p ./taming_transformers/ckpt

echo ""
echo "You need to download the following model weights:"
echo ""
echo "1. Visual Encoder: clip-vit-large-patch14-336"
echo "   Source: https://huggingface.co/openai/clip-vit-large-patch14-336"
echo "   Local Path: ./clip-vit-large-patch14-336/"
echo ""
echo "2. Base LLM Models:"
echo "   - HealthGPT-M3: Phi-3-mini-4k-instruct"
echo "     Source: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct"
echo "     Local Path: ./Phi-3-mini-4k-instruct/"
echo ""
echo "   - HealthGPT-L14: phi-4"
echo "     Source: https://huggingface.co/microsoft/phi-4"
echo "     Local Path: ./phi-4/"
echo ""
echo "3. HealthGPT Weights:"
echo "   Source: https://huggingface.co/lintw/HealthGPT-M3 and https://huggingface.co/lintw/HealthGPT-L14"
echo "   Files Needed:"
echo "     - com_hlora_weights.bin - For comprehension tasks (M3)"
echo "     - gen_hlora_weights.bin - For generation tasks (M3)"
echo "     - fusion_layer_weights.bin - For both tasks (M3)"
echo "     - com_hlora_weights_phi4.bin - For comprehension tasks (L14)"
echo ""
echo "4. VQGAN Weights (For Image Generation):"
echo "   Source: https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/?p=%2F&mode=list"
echo "   Local Path: ./taming_transformers/ckpt/"
echo "   Files Needed:"
echo "     - last.ckpt"
echo "     - model.yaml"
echo ""

read -p "Have you downloaded all the required model weights? (y/n): " downloaded_weights
if [[ "$downloaded_weights" != "y" ]]; then
    echo "Please download the model weights as described in the SETUP_INSTRUCTIONS.md file and run this script again."
    exit 1
fi

# Verify model files exist
echo "Verifying model files..."
missing_files=false

# Check CLIP model
if [ ! -d "./clip-vit-large-patch14-336" ] || [ -z "$(ls -A ./clip-vit-large-patch14-336)" ]; then
    echo "❌ CLIP model files missing"
    missing_files=true
fi

# Check Phi-3-mini
if [ ! -d "./Phi-3-mini-4k-instruct" ] || [ -z "$(ls -A ./Phi-3-mini-4k-instruct)" ]; then
    echo "❌ Phi-3-mini model files missing"
    missing_files=true
fi

# Check HealthGPT-M3 weights
if [ ! -f "./HealthGPT-M3/com_hlora_weights.bin" ] || [ ! -f "./HealthGPT-M3/gen_hlora_weights.bin" ] || [ ! -f "./HealthGPT-M3/fusion_layer_weights.bin" ]; then
    echo "❌ HealthGPT-M3 weight files missing"
    missing_files=true
fi

# Check VQGAN weights
if [ ! -f "./taming_transformers/ckpt/last.ckpt" ] || [ ! -f "./taming_transformers/ckpt/model.yaml" ]; then
    echo "❌ VQGAN weight files missing"
    missing_files=true
fi

if [[ "$missing_files" == "true" ]]; then
    echo "Some model files appear to be missing. Please check and try again."
    read -p "Continue anyway? (y/n): " continue_with_missing
    if [[ "$continue_with_missing" != "y" ]]; then
        echo "Setup aborted."
        exit 1
    fi
fi

# Start the HealthGPT API service
echo ""
echo "Starting HealthGPT API service with Docker Compose..."
docker-compose up -d

# Verify the API is running
echo ""
echo "Verifying API service..."
echo "Waiting for the service to start up..."
sleep 10

response=$(curl -s http://localhost:5000/api/health || echo "Failed to connect")
if [[ "$response" == *"healthy"* ]]; then
    echo "✅ HealthGPT API is running successfully!"
    echo "Your Next.js application is now configured to use the real HealthGPT API."
    echo ""
    echo "Integration complete! You can now use the HealthGPT features in your application."
else
    echo "❌ Could not verify that the HealthGPT API is running."
    echo "Check Docker logs for more information:"
    echo "  docker-compose logs healthgpt-api"
    echo ""
    echo "You may need to wait a bit longer for the service to start up completely."
fi

echo ""
echo "Setup process completed." 