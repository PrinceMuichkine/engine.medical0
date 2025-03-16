#!/bin/bash

# Default settings
PORT=5001
PRELOAD_MODELS=true
USE_PHI4=false
DOWNLOAD_FIRST=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --port)
            PORT="$2"
            shift
            shift
            ;;
        --no-preload)
            PRELOAD_MODELS=false
            shift
            ;;
        --phi4)
            USE_PHI4=true
            shift
            ;;
        --download-models)
            DOWNLOAD_FIRST=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --port PORT       Port to run the server on (default: 5001)"
            echo "  --no-preload      Do not preload models at startup"
            echo "  --phi4            Use Phi-4 model instead of Phi-3"
            echo "  --download-models Download models from HuggingFace before starting"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting HealthGPT API Server with the following settings:"
echo "  Port: $PORT"
echo "  Preload models: $PRELOAD_MODELS"
echo "  Use Phi-4: $USE_PHI4"
echo "  Download models first: $DOWNLOAD_FIRST"

# Make this script executable
chmod +x "$0"

# Make the download script executable
if [ -f "download_models.py" ]; then
    chmod +x "download_models.py"
fi

# Download models if requested
if [ "$DOWNLOAD_FIRST" = true ]; then
    echo "Downloading models from HuggingFace..."
    if [ -f "download_models.py" ]; then
        python download_models.py
        if [ $? -ne 0 ]; then
            echo "Warning: Model download failed, but continuing with server startup."
        fi
    else
        echo "Warning: download_models.py not found. Skipping model download."
    fi
fi

# Construct the command line
CMD="python apy.py --port $PORT"

if [ "$PRELOAD_MODELS" = false ]; then
    CMD="$CMD --no-preload-models"
fi

if [ "$USE_PHI4" = true ]; then
    CMD="$CMD --use-phi4"
fi

# Run the server
echo "Running command: $CMD"
eval "$CMD" 