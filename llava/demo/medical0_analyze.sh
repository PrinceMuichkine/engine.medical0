#!/bin/bash

# medical0_analyze.sh - A wrapper script for HealthGPT with v0-style prompting

# Display usage instructions
function show_help {
    echo "HealthGPT Prompt-based Analysis"
    echo "-------------------------------"
    echo "Usage: $0 [image_path] [model] [task] [analysis_type]"
    echo ""
    echo "Parameters:"
    echo "  image_path      Path to the medical image (required)"
    echo "  model           Model to use: phi3 or phi4 (default: phi3)"
    echo "  task            Type of task: analyze or generate (default: analyze)"
    echo "  analysis_type   Type of analysis to perform (default: general)"
    echo ""
    echo "Analysis Types for 'analyze' task:"
    echo "  general         Comprehensive analysis"
    echo "  modality        Identify imaging modality"
    echo "  anatomy         Map anatomical structures"
    echo "  abnormality     Detect abnormalities"
    echo "  congenital      Identify congenital variations"
    echo "  thoracic        Analyze chest structures"
    echo "  abdominal       Analyze abdominal structures"
    echo "  neuro           Interpret neuroimaging"
    echo "  msk             Examine musculoskeletal structures"
    echo "  genitourinary   Analyze urinary tract"
    echo "  diagnosis       Suggest differential diagnoses"
    echo "  brain_viability Brain viability analysis"
    echo ""
    echo "Analysis Types for 'generate' task:"
    echo "  general         Comprehensive enhancement"
    echo "  clarity         Enhance image clarity"
    echo "  highlight       Highlight abnormalities"
    echo "  structure       Enhance structural boundaries"
    echo "  multi           Enhance multiple structures"
    echo ""
    echo "Examples:"
    echo "  $0 kidney_scan.jpg                       # Basic analysis with default settings"
    echo "  $0 brain_mri.jpg phi4 analyze neuro      # Neuroimaging analysis with Phi-4"
    echo "  $0 chest_xray.jpg phi3 generate clarity  # Enhance clarity with Phi-3"
}

# Check if help is requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Process arguments
IMAGE_PATH=${1}
MODEL=${2:-"phi3"}  # Default to phi3
TASK=${3:-"analyze"}  # Default to analyze
ANALYSIS_TYPE=${4:-"general"}  # Default to general analysis

# Validate required parameters
if [[ -z "$IMAGE_PATH" ]]; then
    echo "Error: Image path is required"
    echo "Use --help for usage information"
    exit 1
fi

# Validate image exists
if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "Error: Image file not found: $IMAGE_PATH"
    exit 1
fi

# Validate model parameter
if [[ "$MODEL" != "phi3" && "$MODEL" != "phi4" ]]; then
    echo "Error: Invalid model. Choose 'phi3' or 'phi4'"
    exit 1
fi

# Validate task parameter
if [[ "$TASK" != "analyze" && "$TASK" != "generate" ]]; then
    echo "Error: Invalid task. Choose 'analyze' or 'generate'"
    exit 1
fi

# Verify analysis types for task
if [[ "$TASK" == "analyze" ]]; then
    valid_analysis=(general modality anatomy abnormality congenital thoracic abdominal neuro msk genitourinary diagnosis brain_viability)
    
    if [[ ! " ${valid_analysis[*]} " =~ " ${ANALYSIS_TYPE} " ]]; then
        echo "Warning: Invalid analysis type for 'analyze' task. Using 'general'"
        ANALYSIS_TYPE="general"
    fi
else  # generate task
    valid_analysis=(general clarity highlight structure multi)
    
    if [[ ! " ${valid_analysis[*]} " =~ " ${ANALYSIS_TYPE} " ]]; then
        echo "Warning: Invalid analysis type for 'generate' task. Using 'general'"
        ANALYSIS_TYPE="general"
    fi
fi

echo "==============================================================="
echo "HealthGPT Medical Image Analysis"
echo "==============================================================="
echo "Image:        $IMAGE_PATH"
echo "Model:        $MODEL (${MODEL/phi3/HealthGPT-M3}${MODEL/phi4/HealthGPT-L14})"
echo "Task:         $TASK"
echo "Analysis:     $ANALYSIS_TYPE"
echo "---------------------------------------------------------------"

# Set output path for generation tasks
if [[ "$TASK" == "generate" ]]; then
    OUTPUT_PATH="${IMAGE_PATH%.*}_enhanced.jpg"
    if [[ "$MODEL" == "phi4" ]]; then
        OUTPUT_PATH="${IMAGE_PATH%.*}_enhanced_phi4.jpg"
    fi
    echo "Output file:  $OUTPUT_PATH"
    echo "---------------------------------------------------------------"
fi

# Determine which script to run
SCRIPT_DIR="$(dirname "$0")"

# For debugging - we'll use this to trace execution
echo "Running script with PID $$"

# Run the appropriate script
if [[ "$MODEL" == "phi3" ]]; then
    if [[ "$TASK" == "analyze" ]]; then
        bash "$SCRIPT_DIR/com_infer.sh" "$IMAGE_PATH" "$ANALYSIS_TYPE"
        SCRIPT_STATUS=$?
    else
        bash "$SCRIPT_DIR/gen_infer.sh" "$IMAGE_PATH" "$OUTPUT_PATH" "$ANALYSIS_TYPE"
        SCRIPT_STATUS=$?
    fi
else  # phi4
    if [[ "$TASK" == "analyze" ]]; then
        bash "$SCRIPT_DIR/com_infer_phi4.sh" "$IMAGE_PATH" "$ANALYSIS_TYPE"
        SCRIPT_STATUS=$?
    else
        bash "$SCRIPT_DIR/gen_infer_phi4.sh" "$IMAGE_PATH" "$OUTPUT_PATH" "$ANALYSIS_TYPE"
        SCRIPT_STATUS=$?
    fi
fi

# Capture exit status from the script we ran
echo "Debug: Script returned status code: $SCRIPT_STATUS"

# Print final status message based on script exit code
if [ $SCRIPT_STATUS -eq 0 ]; then
    echo "==============================================================="
    echo "✅ Analysis completed successfully!"
    
    if [[ "$TASK" == "generate" ]]; then
        echo "Enhanced image saved to: $OUTPUT_PATH"
    fi
    
    echo "==============================================================="
    exit 0
else
    echo "==============================================================="
    echo "❌ Analysis failed with exit code: $SCRIPT_STATUS"
    echo "Please check the error messages above for details."
    echo "==============================================================="
    # Exit with the same error code that the script returned
    exit $SCRIPT_STATUS
fi 