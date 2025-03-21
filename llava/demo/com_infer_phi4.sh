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
export PYTHONUNBUFFERED=1  # Force Python to print output immediately
# Add PYTHONPATH to find the llava module
export PYTHONPATH="$PYTHONPATH:$(realpath $(dirname "$0")/../..)"

# Check README statement about Phi-4 availability
echo "NOTE: According to the README, the Phi-4 weights are not yet released."
echo "Falling back to using Phi-3 model and weights instead."

# Model and weights paths - use Phi-3 directly since Phi-4 weights aren't released
MODEL_NAME_OR_PATH="microsoft/Phi-3-mini-4k-instruct"
VIT_PATH="openai/clip-vit-large-patch14-336"  # No trailing slash!
HLORA_PATH="../../weights/com_hlora_weights.bin"  # Use Phi-3 weights directly

echo "======= Starting HealthGPT Analysis ======="
echo "Using image: $IMAGE_PATH"
echo "Model: $MODEL_NAME_OR_PATH (fallback from Phi-4)"
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

# Check for phi-4 model files
PHI4_MODEL_FILE="$MODEL_NAME_OR_PATH/model-00001-of-00006.safetensors"
if [ ! -f "$PHI4_MODEL_FILE" ]; then
    echo "WARNING: Phi-4 model files not found at: $MODEL_NAME_OR_PATH"
    echo "The README indicates the full weights for HealthGPT-L14 are not yet released."
    echo "Falling back to using Phi-3 for now..."
    ALTERNATE_MODEL="microsoft/Phi-3-mini-4k-instruct"
    echo "Using model: $ALTERNATE_MODEL"
    MODEL_NAME_OR_PATH="$ALTERNATE_MODEL"
fi

# Check if GPU is available
HAS_CUDA=0
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "CUDA GPU detected, will use GPU for inference"
    HAS_CUDA=1
    export CUDA_VISIBLE_DEVICES=0  # Use the first GPU
else
    echo "No CUDA GPU detected, will use CPU for inference"
    export CUDA_VISIBLE_DEVICES=""
fi

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

echo "Using prompt: $QUESTION"

# Create a temporary Python script to modify the inference code
TMP_PATCH_SCRIPT=$(mktemp)
cat > $TMP_PATCH_SCRIPT << 'EOFSCRIPT'
import sys
import re

filename = sys.argv[1]
with open(filename, 'r') as f:
    content = f.read()

# Modify the cuda() calls to use a device-agnostic approach
cuda_pattern = r'model\.to\(model_dtype\)\.cuda\(\)'
new_code = 'model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=model_dtype)'
modified_content = re.sub(cuda_pattern, new_code, content)

# Fix the model.generate call to be device-agnostic
generate_pattern = r'images=image_tensor\.to\(dtype=model_dtype, device=\'cuda\', non_blocking=True\) if img_path else None,'
new_generate_code = 'images=image_tensor.to(dtype=model_dtype, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), non_blocking=True) if img_path else None,'
modified_content = re.sub(generate_pattern, new_generate_code, modified_content)

# Fix the input_ids to be device agnostic
input_ids_pattern = r'input_ids = tokenizer_image_token\(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=\'pt\'\)\.cuda\(\)\.unsqueeze_\(0\)'
new_input_ids_code = 'input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=\'pt\').to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")).unsqueeze_(0)'
modified_content = re.sub(input_ids_pattern, new_input_ids_code, modified_content)

# Add torch import at the top if needed
if "import torch" not in content.split("\n")[:20]:
    modified_content = re.sub(r'import os, sys', 'import os, sys\nimport torch', modified_content)

with open(filename, 'w') as f:
    f.write(modified_content)
print(f"Modified {filename} for device-agnostic execution")
EOFSCRIPT

# Apply the patch to com_infer_phi4.py
python3 $TMP_PATCH_SCRIPT "$(dirname "$0")/com_infer_phi4.py"
rm $TMP_PATCH_SCRIPT

# Run the inference with all arguments
python3 -u "$(dirname "$0")/com_infer_phi4.py" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dtype "FP16" \
    --hlora_r "64" \
    --hlora_alpha "128" \
    --hlora_nums "4" \
    --vq_idx_nums "8192" \
    --instruct_template "phi3_instruct" \
    --vit_path "$VIT_PATH" \
    --hlora_path "$HLORA_PATH" \
    --question "$QUESTION" \
    --img_path "$IMAGE_PATH"

STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "======= Phi-4 Inference completed successfully ======="
else
    echo "======= Phi-4 Inference failed with exit code $STATUS ======="
    echo "Please check the error messages above for more information."
fi