#!/bin/bash

# Get the image path and output path from command line arguments
IMAGE_PATH=${1:-"path/to/default/image.jpg"}
OUTPUT_PATH=${2:-"output.jpg"}
PROMPT_TYPE=${3:-"general"}

# Activate conda environment to ensure all dependencies are available
if [[ -f "/root/anaconda3/etc/profile.d/conda.sh" ]]; then
    echo "Activating conda environment..."
    source "/root/anaconda3/etc/profile.d/conda.sh"
    conda activate HealthGPT
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    echo "Activating conda environment from /opt/conda..."
    source "/opt/conda/etc/profile.d/conda.sh"
    conda activate HealthGPT
else
    echo "WARNING: Could not find conda.sh to activate environment"
    if [[ -d "/opt/conda/envs/HealthGPT" ]]; then
        echo "Trying direct activation of environment..."
        export PATH="/opt/conda/envs/HealthGPT/bin:$PATH"
    fi
fi

MODEL_NAME_OR_PATH="microsoft/Phi-3-mini-4k-instruct"
VIT_PATH="openai/clip-vit-large-patch14-336"  # No trailing slash!
HLORA_PATH="../../weights/gen_hlora_weights.bin"
FUSION_LAYER_PATH="../../weights/fusion_layer_weights.bin"

# New code to read prompts from files
PROMPT_FILE="../../medical0.tools.txt"
if [ -f "$PROMPT_FILE" ] && [ "$PROMPT_TYPE" != "general" ]; then
    # Extract different prompt types based on parameter
    case "$PROMPT_TYPE" in
        "clarity")
            QUESTION=$(grep -A 10 "a) Clarity Enhancement:" "$PROMPT_FILE" | grep "Example prompt:" | sed 's/.*Example prompt: "\([^"]*\)".*/\1/')
            ;;
        "highlight")
            QUESTION=$(grep -A 10 "b) Abnormality Highlighting:" "$PROMPT_FILE" | grep "Example prompt:" | sed 's/.*Example prompt: "\([^"]*\)".*/\1/')
            ;;
        "structure")
            QUESTION=$(grep -A 10 "c) Structural Delineation:" "$PROMPT_FILE" | grep "Example prompt:" | sed 's/.*Example prompt: "\([^"]*\)".*/\1/')
            ;;
        "multi")
            QUESTION=$(grep -A 10 "d) Multi-structure Enhancement:" "$PROMPT_FILE" | grep "Example prompt:" | sed 's/.*Example prompt: "\([^"]*\)".*/\1/')
            ;;
        *)
            # Default comprehensive enhancement
            QUESTION="Reconstruct this medical image with enhanced clarity. Highlight any anatomical abnormalities, congenital variations, or pathological findings. Pay special attention to organ position, structural relationships, and any asymmetries. Make sure to include the location of the abnormalities."
            ;;
    esac
else
    # Default prompt if file not found
    QUESTION="Reconstruct this medical image with enhanced clarity. Highlight any anatomical abnormalities, congenital variations, or pathological findings. Pay special attention to organ position, structural relationships, and any asymmetries. Make sure to include the location of the abnormalities."
fi

echo "Using prompt: $QUESTION"

python3 gen_infer.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dtype "FP16" \
    --hlora_r "256" \
    --hlora_alpha "512" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi3_instruct" \
    --vit_path "$VIT_PATH" \
    --hlora_path "$HLORA_PATH" \
    --fusion_layer_path "$FUSION_LAYER_PATH" \
    --question "$QUESTION" \
    --img_path "$IMAGE_PATH" \
    --save_path "$OUTPUT_PATH"