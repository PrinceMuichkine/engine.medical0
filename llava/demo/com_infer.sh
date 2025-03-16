#!/bin/bash

# Get the image path from command line argument
IMAGE_PATH=${1:-"path/to/default/image.jpg"}
PROMPT_TYPE=${2:-"general"}
USE_CACHE=${3:-"true"}  # New parameter to control caching

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

# Model and weights paths - default location
MODEL_NAME_OR_PATH="microsoft/Phi-3-mini-4k-instruct"
VIT_PATH="openai/clip-vit-large-patch14-336"
HLORA_PATH="../../weights/com_hlora_weights.bin"
FUSION_LAYER_PATH="../../weights/fusion_layer_weights.bin"

# Alternative weight paths to try if the default isn't found
ALT_WEIGHT_PATHS=(
    "../../weights" # Default relative path
    "./weights"     # Local weights directory
    "../weights"    # Parent directory
    "weights"       # Current directory
    "/root/engine.medical0/weights" # Absolute path to project root
)

# Function to find a file in multiple locations
find_file() {
    local filename=$1
    local default_path=$2
    
    # First check if the default path exists
    if [ -f "$default_path" ]; then
        echo "$default_path"
        return 0
    fi
    
    # Try alternative paths
    for path in "${ALT_WEIGHT_PATHS[@]}"; do
        local full_path="$path/$(basename "$filename")"
        if [ -f "$full_path" ]; then
            echo "$full_path"
            return 0
        fi
    done
    
    # Fall back to the default if nothing is found
    echo "$default_path"
    return 1
}

# Find weight files
HLORA_PATH=$(find_file "com_hlora_weights.bin" "$HLORA_PATH")
FUSION_LAYER_PATH=$(find_file "fusion_layer_weights.bin" "$FUSION_LAYER_PATH")

# Print the paths being used
echo "Using HLORA weights path: $HLORA_PATH"
echo "Using fusion layer path: $FUSION_LAYER_PATH"

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
echo "Caching enabled: $USE_CACHE"

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "ERROR: Image file does not exist: $IMAGE_PATH"
    exit 1
fi

# Advanced weight file handling
check_and_resolve_weight_file() {
    local file_path=$1
    local file_name=$(basename "$file_path")
    local source_path=""
    
    if [ -f "$file_path" ]; then
        echo "Found $file_name at $file_path"
        return 0
    fi
    
    echo "WARNING: $file_name not found at expected location: $file_path"
    
    # Try to find it in common locations
    for search_dir in "/root/engine.medical0/weights" "/weights" "./weights" "../weights" "../../weights"; do
        test_path="$search_dir/$file_name"
        if [ -f "$test_path" ]; then
            source_path="$test_path"
            echo "Found $file_name at alternative location: $source_path"
            break
        fi
    done
    
    # If found elsewhere, try to create a symlink or copy it
    if [ -n "$source_path" ]; then
        # Create the target directory if it doesn't exist
        mkdir -p "$(dirname "$file_path")"
        
        # Try to create a symlink first
        echo "Creating symlink from $source_path to $file_path"
        ln -sf "$source_path" "$file_path" 2>/dev/null
        
        # If symlink failed, try to copy
        if [ ! -f "$file_path" ]; then
            echo "Symlink failed, copying file instead"
            cp "$source_path" "$file_path"
        fi
        
        # Check if it worked
        if [ -f "$file_path" ]; then
            echo "Successfully made $file_name available at $file_path"
            return 0
        fi
    fi
    
    # Final absolute path check - use it directly if found
    for abs_path in $(find /root -name "$file_name" 2>/dev/null); do
        echo "Found $file_name at: $abs_path"
        echo "Using absolute path instead of relative path"
        # Override the path variable in the parent scope
        eval "$2=\"$abs_path\""
        return 0
    done
    
    echo "ERROR: Could not locate $file_name in any standard location"
    return 1
}

# Check weights and try to resolve issues
check_and_resolve_weight_file "$HLORA_PATH" "HLORA_PATH" || {
    echo "ERROR: Failed to locate or access H-LoRA weights file"
    exit 1
}

check_and_resolve_weight_file "$FUSION_LAYER_PATH" "FUSION_LAYER_PATH" || {
    echo "ERROR: Failed to locate or access fusion layer weights file"
    exit 1
}

# Print execution trace
set -x

# Run the inference with all arguments
echo "Running model inference..."
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
    --img_path "$IMAGE_PATH" \
    --use_cache "$USE_CACHE"

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
            --img_path "$IMAGE_PATH" \
            --use_cache "$USE_CACHE"
            
        if [ $? -eq 0 ]; then
            echo "======= CPU inference completed successfully ======="
        else
            echo "======= CPU inference also failed ======="
        fi
    fi
fi