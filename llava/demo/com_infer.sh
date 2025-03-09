#!/bin/bash

# Get the image path from command line argument
IMAGE_PATH=${1:-"path/to/default/image.jpg"}
PROMPT_TYPE=${2:-"general"}

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

# Set environment variables
export PYTHONUNBUFFERED=1

# Model and weights paths
MODEL_NAME_OR_PATH="microsoft/Phi-3-mini-4k-instruct"
VIT_PATH="openai/clip-vit-large-patch14-336"
HLORA_PATH="../../weights/com_hlora_weights.bin"
FUSION_LAYER_PATH="../../weights/fusion_layer_weights.bin"

# New code to read prompts from files
PROMPT_FILE="../../medical0.tools.txt"
if [ -f "$PROMPT_FILE" ] && [ "$PROMPT_TYPE" != "general" ]; then
    # Extract different prompt types based on parameter
    case "$PROMPT_TYPE" in
        "modality")
            QUESTION=$(grep -A 10 "a) Modality Recognition:" "$PROMPT_FILE" | grep "Example prompt:" | sed 's/.*Example prompt: "\([^"]*\)".*/\1/')
            ;;
        "anatomy")
            QUESTION=$(grep -A 10 "b) Anatomical Mapping:" "$PROMPT_FILE" | grep "Example prompt:" | sed 's/.*Example prompt: "\([^"]*\)".*/\1/')
            ;;
        "abnormality")
            QUESTION=$(grep -A 10 "c) Abnormality Detection:" "$PROMPT_FILE" | grep "Example prompt:" | sed 's/.*Example prompt: "\([^"]*\)".*/\1/')
            ;;
        "congenital")
            QUESTION=$(grep -A 10 "d) Congenital Variant Recognition:" "$PROMPT_FILE" | grep "Example prompt:" | sed 's/.*Example prompt: "\([^"]*\)".*/\1/')
            ;;
        "thoracic")
            QUESTION=$(grep -A 10 "a) Thoracic Analysis:" "$PROMPT_FILE" | grep "Example prompt:" | sed 's/.*Example prompt: "\([^"]*\)".*/\1/')
            ;;
        "abdominal")
            QUESTION=$(grep -A 10 "b) Abdominal Assessment:" "$PROMPT_FILE" | grep "Example prompt:" | sed 's/.*Example prompt: "\([^"]*\)".*/\1/')
            ;;
        "neuro")
            QUESTION=$(grep -A 10 "c) Neuroimaging Interpretation:" "$PROMPT_FILE" | grep "Example prompt:" | sed 's/.*Example prompt: "\([^"]*\)".*/\1/')
            ;;
        "brain_viability")
            QUESTION=$(grep -A 10 "f) Brain Viability Assessment:" "$PROMPT_FILE" | grep "Example prompt:" | sed 's/.*Example prompt: "\([^"]*\)".*/\1/')
            ;;
        "msk")
            QUESTION=$(grep -A 10 "d) Musculoskeletal Examination:" "$PROMPT_FILE" | grep "Example prompt:" | sed 's/.*Example prompt: "\([^"]*\)".*/\1/')
            ;;
        "genitourinary")
            QUESTION=$(grep -A 10 "e) Genitourinary System Analysis:" "$PROMPT_FILE" | grep "Example prompt:" | sed 's/.*Example prompt: "\([^"]*\)".*/\1/')
            ;;
        "diagnosis")
            QUESTION=$(grep -A 10 "a) Differential Diagnosis:" "$PROMPT_FILE" | grep "Example prompt:" | sed 's/.*Example prompt: "\([^"]*\)".*/\1/')
            ;;
        *)
            # Default comprehensive analysis
            QUESTION="Analyze this medical image. Identify the imaging modality, describe visible anatomical structures, and note any abnormalities, congenital variations, or developmental anomalies. Include observations about organ position, shape, and symmetry."
            ;;
    esac
else
    # Default prompt if file not found
    QUESTION="Analyze this medical image. Identify the imaging modality, describe visible anatomical structures, and note any abnormalities, congenital variations, or developmental anomalies. Include observations about organ position, shape, and symmetry."
fi

echo "======= Starting HealthGPT-M3 Inference ======="
echo "Using image: $IMAGE_PATH"
echo "Using prompt: $QUESTION"
echo "Model: $MODEL_NAME_OR_PATH"
echo "VIT path: $VIT_PATH"

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "ERROR: Image file does not exist: $IMAGE_PATH"
    exit 1
fi

# Check if weights exist
if [ ! -f "$HLORA_PATH" ]; then
    echo "ERROR: H-LoRA weights file does not exist: $HLORA_PATH"
    exit 1
fi

if [ ! -f "$FUSION_LAYER_PATH" ]; then
    echo "ERROR: Fusion layer weights file does not exist: $FUSION_LAYER_PATH"
    exit 1
fi

# Print execution trace
set -x

# Run the inference with all arguments
python3 -u com_infer.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dtype "FP16" \
    --hlora_r "64" \
    --hlora_alpha "128" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi3_instruct" \
    --vit_path "$VIT_PATH" \
    --hlora_path "$HLORA_PATH" \
    --fusion_layer_path "$FUSION_LAYER_PATH" \
    --question "$QUESTION" \
    --img_path "$IMAGE_PATH"

# Capture the exit status
STATUS=$?

# Turn off trace
set +x

if [ $STATUS -eq 0 ]; then
    echo "======= Inference completed successfully ======="
else
    echo "======= Inference failed with exit code $STATUS ======="
    
    # If it failed, try with CPU as fallback
    if [ "$STATUS" -ne 0 ]; then
        echo "Attempting fallback to CPU execution..."
        export CUDA_VISIBLE_DEVICES=""
        
        python3 -u com_infer.py \
            --model_name_or_path "$MODEL_NAME_OR_PATH" \
            --dtype "FP16" \
            --hlora_r "64" \
            --hlora_alpha "128" \
            --hlora_nums "4" \
            --vq_idx_nums "8192" \
            --instruct_template "phi3_instruct" \
            --vit_path "$VIT_PATH" \
            --hlora_path "$HLORA_PATH" \
            --fusion_layer_path "$FUSION_LAYER_PATH" \
            --question "$QUESTION" \
            --img_path "$IMAGE_PATH"
            
        if [ $? -eq 0 ]; then
            echo "======= CPU inference completed successfully ======="
        else
            echo "======= CPU inference also failed ======="
        fi
    fi
fi