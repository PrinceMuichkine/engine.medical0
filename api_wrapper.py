#!/usr/bin/env python3
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import os
import sys
import json
import traceback
import argparse

# Add the current directory to the path to access llava modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the necessary modules
try:
    import torch
    # Check what demo scripts are available
    demo_dir = os.path.join(current_dir, "llava", "demo")
    com_script_path = os.path.join(demo_dir, "com_infer.py")
    gen_script_path = os.path.join(demo_dir, "gen_infer.py")
    
    # If the files don't exist, adjust paths for phi4 versions
    if not os.path.exists(com_script_path):
        com_script_path = os.path.join(demo_dir, "com_infer_phi4.py")
    if not os.path.exists(gen_script_path):
        gen_script_path = os.path.join(demo_dir, "gen_infer_phi4.py")
        
    print(f"Found comprehension script: {com_script_path}")
    print(f"Found generation script: {gen_script_path}")
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# Define model configurations
configs = {
    "HealthGPT-M3": {
        "model_name_or_path": "./Phi-3-mini-4k-instruct",
        "dtype": "FP16",
        "hlora_r": 64,
        "hlora_alpha": 128,
        "hlora_nums": 4,
        "vq_idx_nums": 8192,
        "instruct_template": "phi3_instruct",
        "vit_path": "./clip-vit-large-patch14-336",
        "hlora_path": "./HealthGPT-M3/com_hlora_weights.bin",
        "fusion_layer_path": "./HealthGPT-M3/fusion_layer_weights.bin"
    },
    "HealthGPT-M3-Gen": {
        "model_name_or_path": "./Phi-3-mini-4k-instruct",
        "dtype": "FP16", 
        "hlora_r": 256,
        "hlora_alpha": 512,
        "hlora_nums": 4,
        "vq_idx_nums": 8192,
        "instruct_template": "phi3_instruct",
        "vit_path": "./clip-vit-large-patch14-336",
        "hlora_path": "./HealthGPT-M3/gen_hlora_weights.bin", 
        "fusion_layer_path": "./HealthGPT-M3/fusion_layer_weights.bin"
    }
}

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image."""
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def run_script(script_path, args_dict):
    """Run a Python script with the provided arguments"""
    import subprocess
    
    # Convert dictionary to command-line arguments
    cmd_args = [sys.executable, script_path]
    for key, value in args_dict.items():
        if value is not None:
            cmd_args.append(f"--{key}")
            cmd_args.append(str(value))
    
    print(f"Running command: {' '.join(cmd_args)}")
    
    # Run the script and capture output
    process = subprocess.Popen(
        cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error running script: {stderr}")
        raise Exception(f"Script execution failed: {stderr}")
    
    return stdout

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if the API is running."""
    return jsonify({
        "status": "healthy",
        "gpu": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze an image using the comprehension model."""
    try:
        # Get data from request
        data = request.json
        image_base64 = data.get('image', '')
        prompt = data.get('prompt', '')
        model_type = data.get('model', 'HealthGPT-M3')
        
        # Basic validation
        if not image_base64:
            return jsonify({"status": "error", "message": "No image provided"})
        
        # Convert base64 to image and save to temp file
        image = base64_to_image(image_base64)
        temp_img_path = os.path.join(current_dir, "temp_input_image.jpg")
        image.save(temp_img_path)
        
        # Get model config
        if model_type not in configs:
            return jsonify({"status": "error", "message": f"Model {model_type} not found"})
        model_config = configs[model_type]
        
        # Prepare script arguments
        script_args = {
            "model_name_or_path": model_config["model_name_or_path"],
            "dtype": model_config["dtype"],
            "hlora_r": model_config["hlora_r"],
            "hlora_alpha": model_config["hlora_alpha"],
            "hlora_nums": model_config["hlora_nums"],
            "vq_idx_nums": model_config["vq_idx_nums"],
            "instruct_template": model_config["instruct_template"],
            "vit_path": model_config["vit_path"],
            "hlora_path": model_config["hlora_path"],
            "fusion_layer_path": model_config["fusion_layer_path"],
            "question": prompt,
            "img_path": temp_img_path
        }
        
        # Run the comprehension script
        output = run_script(com_script_path, script_args)
        
        # Extract the response from the output
        response = ""
        for line in output.splitlines():
            if line.startswith("HealthGPT:"):
                response = line.replace("HealthGPT:", "").strip()
                break
        
        # Clean up temp file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        
        # Return the result
        return jsonify({
            "result": response,
            "status": "success"
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/generate', methods=['POST'])
def generate_image():
    """Generate an image using the generation model."""
    try:
        # Get data from request
        data = request.json
        image_base64 = data.get('image', '')
        prompt = data.get('prompt', '')
        
        # Basic validation
        if not image_base64:
            return jsonify({"status": "error", "message": "No image provided"})
        
        # Convert base64 to image and save to temp file
        image = base64_to_image(image_base64)
        temp_img_path = os.path.join(current_dir, "temp_input_image.jpg")
        temp_out_path = os.path.join(current_dir, "temp_output_image.jpg")
        image.save(temp_img_path)
        
        # Get model config for generation
        model_config = configs["HealthGPT-M3-Gen"]
        
        # Prepare script arguments
        script_args = {
            "model_name_or_path": model_config["model_name_or_path"],
            "dtype": model_config["dtype"],
            "hlora_r": model_config["hlora_r"],
            "hlora_alpha": model_config["hlora_alpha"],
            "hlora_nums": model_config["hlora_nums"],
            "vq_idx_nums": model_config["vq_idx_nums"],
            "instruct_template": model_config["instruct_template"],
            "vit_path": model_config["vit_path"],
            "hlora_path": model_config["hlora_path"],
            "fusion_layer_path": model_config["fusion_layer_path"],
            "question": prompt,
            "img_path": temp_img_path,
            "save_path": temp_out_path
        }
        
        # Run the generation script
        run_script(gen_script_path, script_args)
        
        # Read the generated image
        if os.path.exists(temp_out_path):
            generated_image = Image.open(temp_out_path)
            img_base64 = f"data:image/png;base64,{image_to_base64(generated_image)}"
            
            # Clean up
            os.remove(temp_out_path)
        else:
            return jsonify({"status": "error", "message": "Image generation failed"})
        
        # Clean up temp file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        
        # Return the result
        return jsonify({
            "image": img_base64,
            "status": "success"
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get information about available models."""
    return jsonify({
        "models": [
            {
                "id": "HealthGPT-M3", 
                "name": "HealthGPT-M3", 
                "description": "Smaller model optimized for speed"
            }
        ]
    })

if __name__ == '__main__':
    print("Starting HealthGPT API...")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    app.run(host='0.0.0.0', port=5000) 