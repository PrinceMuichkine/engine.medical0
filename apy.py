import os
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import base64
from io import BytesIO
from PIL import Image
import uuid
import re
import numpy as np
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('healthgpt-api')

# Set custom temp directory
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Available analysis types for UI display
ANALYSIS_TYPES = {
    "analyze": [
        {"id": "general", "name": "General Analysis", "description": "Comprehensive medical image analysis"},
        {"id": "modality", "name": "Modality Identification", "description": "Identify the imaging technique used"},
        {"id": "anatomy", "name": "Anatomical Mapping", "description": "Map visible anatomical structures"},
        {"id": "abnormality", "name": "Abnormality Detection", "description": "Detect abnormalities and pathologies"},
        {"id": "congenital", "name": "Congenital Variant Recognition", "description": "Identify developmental anomalies"},
        {"id": "thoracic", "name": "Thoracic Analysis", "description": "Detailed assessment of chest imaging"},
        {"id": "abdominal", "name": "Abdominal Assessment", "description": "Analysis of abdominal structures"},
        {"id": "neuro", "name": "Neuroimaging Interpretation", "description": "Brain and spinal cord analysis"},
        {"id": "msk", "name": "Musculoskeletal Examination", "description": "Assessment of bones, joints, and tissues"},
        {"id": "genitourinary", "name": "Genitourinary Analysis", "description": "Urinary tract and reproductive organs"}
    ],
    "generate": [
        {"id": "general", "name": "General Enhancement", "description": "Comprehensive image enhancement"},
        {"id": "clarity", "name": "Clarity Enhancement", "description": "Improve image visibility"},
        {"id": "highlight", "name": "Abnormality Highlighting", "description": "Highlight areas of pathology"},
        {"id": "structure", "name": "Structural Delineation", "description": "Define boundaries between structures"},
        {"id": "multi", "name": "Multi-structure Enhancement", "description": "Enhance multiple anatomical features"}
    ]
}

# Mapping of keywords to analysis types
KEYWORD_MAPPING = {
    "analyze": {
        "brain": "neuro",
        "cerebr": "neuro",
        "skull": "neuro",
        "head": "neuro",
        "neural": "neuro",
        "ventricle": "neuro",
        
        "chest": "thoracic",
        "thorax": "thoracic",
        "lung": "thoracic",
        "heart": "thoracic",
        "cardiac": "thoracic",
        "pulmonary": "thoracic",
        "mediastin": "thoracic",
        
        "abdomen": "abdominal",
        "liver": "abdominal",
        "spleen": "abdominal",
        "pancreas": "abdominal",
        "bowel": "abdominal",
        "colon": "abdominal",
        "intestine": "abdominal",
        "stomach": "abdominal",
        "gallbladder": "abdominal",
        
        "bone": "msk",
        "joint": "msk",
        "fracture": "msk",
        "orthopedic": "msk",
        "tendon": "msk",
        "muscle": "msk",
        "skeletal": "msk",
        
        "kidney": "genitourinary",
        "bladder": "genitourinary",
        "urinary": "genitourinary",
        "renal": "genitourinary",
        "prostate": "genitourinary",
        "uterus": "genitourinary",
        "ovary": "genitourinary",
        "ureter": "genitourinary",
        
        "modality": "modality",
        "imaging": "modality",
        "technique": "modality",
        "protocol": "modality",
        
        "anatomy": "anatomy",
        "structure": "anatomy",
        "organ": "anatomy",
        
        "abnormal": "abnormality",
        "pathology": "abnormality",
        "lesion": "abnormality",
        "mass": "abnormality",
        "tumor": "abnormality",
        "cancer": "abnormality",
        
        "congenital": "congenital",
        "developmental": "congenital",
        "anomaly": "congenital",
        "variant": "congenital",
        "variation": "congenital",
        "malformation": "congenital",
        "horseshoe": "congenital"
    },
    "generate": {
        "clarity": "clarity",
        "clear": "clarity",
        "enhance": "clarity",
        "sharpen": "clarity",
        
        "highlight": "highlight",
        "mark": "highlight",
        "indicate": "highlight",
        "show": "highlight",
        "point": "highlight",
        
        "structure": "structure",
        "border": "structure",
        "boundary": "structure",
        "edge": "structure",
        "outline": "structure",
        
        "all": "multi",
        "multiple": "multi",
        "several": "multi",
        "various": "multi",
        "overall": "multi"
    }
}

def extract_prompt_from_tools_file(analysis_type, task="analyze"):
    """Extract specific prompt from medical0.tools.txt based on analysis type"""
    tools_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "medical0.tools.txt")
    
    if not os.path.exists(tools_file):
        return None
        
    try:
        with open(tools_file, "r") as f:
            content = f.read()
            
        # Different sections to search based on analysis type and task
        search_patterns = {
            "analyze": {
                "modality": "a\\) Modality Recognition:",
                "anatomy": "b\\) Anatomical Mapping:",
                "abnormality": "c\\) Abnormality Detection:",
                "congenital": "d\\) Congenital Variant Recognition:",
                "thoracic": "a\\) Thoracic Analysis:",
                "abdominal": "b\\) Abdominal Assessment:",
                "neuro": "c\\) Neuroimaging Interpretation:",
                "msk": "d\\) Musculoskeletal Examination:",
                "genitourinary": "e\\) Genitourinary System Analysis:",
                "diagnosis": "a\\) Differential Diagnosis:"
            },
            "generate": {
                "clarity": "a\\) Clarity Enhancement:",
                "highlight": "b\\) Abnormality Highlighting:",
                "structure": "c\\) Structural Delineation:",
                "multi": "d\\) Multi-structure Enhancement:"
            }
        }
        
        # Find the relevant pattern
        pattern = search_patterns.get(task, {}).get(analysis_type)
        
        if pattern:
            # Search for the example prompt
            section_match = re.search(f"{pattern}.*?Example prompt: \"(.*?)\"", content, re.DOTALL)
            if section_match:
                return section_match.group(1)
                
    except Exception as e:
        logger.error(f"Error extracting prompt: {str(e)}")
        
    return None

def detect_analysis_type(image_data=None, question=None, task="analyze"):
    """
    Auto-detect appropriate analysis type based on image and/or question
    
    Args:
        image_data: Raw image data (bytes)
        question: User's question or prompt (string)
        task: Either "analyze" or "generate"
        
    Returns:
        String representing the analysis type
    """
    # First try to detect from question if provided
    if question:
        question_lower = question.lower()
        
        # Get mapping for this task
        mapping = KEYWORD_MAPPING.get(task, {})
        
        # Check each keyword
        for keyword, analysis_type in mapping.items():
            if keyword in question_lower:
                logger.info(f"Detected analysis type '{analysis_type}' from keyword '{keyword}' in question")
                return analysis_type
    
    # If we have image data, try basic image analysis for detection
    if image_data and task == "analyze":
        try:
            # Convert image bytes to cv2 image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Failed to decode image")
                return "general"
                
            # Get image properties
            height, width, _ = img.shape
            aspect_ratio = width / height
            
            # Perform some basic detection based on image dimensions
            if 0.9 <= aspect_ratio <= 1.1:  # Approximately square
                if width >= 512:  # Higher resolution might be CT or MRI
                    logger.info("Image appears to be a cross-sectional scan (CT/MRI), suggesting abdominal/thoracic/neuro")
                    return "general"  # Default to general for cross-sectional images
                else:
                    return "general"
            elif aspect_ratio > 1.3:  # Wide image
                logger.info("Image appears to be a wide format, possibly chest X-ray")
                return "thoracic"
            elif aspect_ratio < 0.8:  # Tall image
                logger.info("Image appears to be a vertical format, possibly spine or long bone")
                return "msk"
        except Exception as e:
            logger.error(f"Error in image-based detection: {str(e)}")
    
    # Default to general analysis
    return "general"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "HealthGPT API is running"}), 200

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """Endpoint for medical image analysis"""
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "Missing image data"}), 400
    
    # Optional parameters
    analysis_type = data.get('analysis_type')  # Now optional
    question = data.get('question')  # Still optional
    use_phi4 = data.get('use_phi4', False)  # Default to Phi-3 unless specified
    
    try:
        # Decode image
        image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
        image_path = os.path.join(temp_dir, f"upload_{uuid.uuid4()}.jpg")
        
        with open(image_path, "wb") as f:
            f.write(image_data)
        
        # Auto-detect analysis type if not provided
        if not analysis_type:
            analysis_type = detect_analysis_type(image_data, question, "analyze")
            logger.info(f"Auto-detected analysis type: {analysis_type}")
        
        # If no question is provided, try to get one from the medical0.tools.txt file
        if not question:
            # Try to get question from tools file first
            question = extract_prompt_from_tools_file(analysis_type, "analyze")
            
            # Default fallback if not found
            if not question:
                question = "Analyze this medical image. Identify the imaging modality, describe visible anatomical structures, and note any abnormalities, congenital variations, or developmental anomalies. Include observations about organ position, shape, and symmetry."
        
        # Use our wrapper script if it exists, otherwise fall back to original approach
        wrapper_script = os.path.join("llava", "demo", "medical0_analyze.sh")
        if os.path.exists(wrapper_script):
            # Use the wrapper script with all parameters
            model_param = "phi4" if use_phi4 else "phi3"
            output = subprocess.check_output(
                [wrapper_script, image_path, model_param, "analyze", analysis_type], 
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
        else:
            # Original approach: Choose the appropriate script
            script_path = os.path.join("llava", "demo", "com_infer_phi4.sh" if use_phi4 else "com_infer.sh")
            
            # Run the script with the analysis type
            output = subprocess.check_output(
                [script_path, image_path, analysis_type], 
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
        
        # Extract the answer - everything after "HealthGPT: "
        if "HealthGPT: " in output:
            answer = output.split("HealthGPT: ")[1].strip()
        else:
            answer = "Unable to extract the response from the model output."
        
        result = {
            "answer": answer, 
            "model": "Phi-4" if use_phi4 else "Phi-3-mini",
            "analysis_type": analysis_type,
            "prompt": question,
            "detected": analysis_type != data.get('analysis_type') if 'analysis_type' in data else True  # Indicate if this was auto-detected
        }
        
    except subprocess.CalledProcessError as e:
        result = {"error": f"Process error: {e.output}"}
    except Exception as e:
        result = {"error": f"Error processing request: {str(e)}"}
    finally:
        # Clean up
        if os.path.exists(image_path):
            os.remove(image_path)
    
    return jsonify(result)

@app.route('/reconstruct-image', methods=['POST'])
def reconstruct_image():
    """Endpoint for image reconstruction"""
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "Missing image data"}), 400
    
    # Optional parameters
    analysis_type = data.get('analysis_type')  # Now optional
    question = data.get('question')  # Still optional
    use_phi4 = data.get('use_phi4', False)  # Default to Phi-3 unless specified
    
    try:
        # Decode image
        image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
        image_path = os.path.join(temp_dir, f"upload_{uuid.uuid4()}.jpg")
        output_path = os.path.join(temp_dir, f"output_{uuid.uuid4()}.jpg")
        
        with open(image_path, "wb") as f:
            f.write(image_data)
        
        # Auto-detect analysis type if not provided
        if not analysis_type:
            analysis_type = detect_analysis_type(image_data, question, "generate")
            logger.info(f"Auto-detected generation type: {analysis_type}")
        
        # If no question is provided, try to get one from the medical0.tools.txt file
        if not question:
            # Try to get question from tools file first
            question = extract_prompt_from_tools_file(analysis_type, "generate")
            
            # Default fallback if not found
            if not question:
                question = "Reconstruct this medical image with enhanced clarity. Highlight any anatomical abnormalities, congenital variations, or pathological findings. Pay special attention to organ position, structural relationships, and any asymmetries."
        
        # Use our wrapper script if it exists, otherwise fall back to original approach
        wrapper_script = os.path.join("llava", "demo", "medical0_analyze.sh")
        if os.path.exists(wrapper_script):
            # Use the wrapper script with all parameters
            model_param = "phi4" if use_phi4 else "phi3"
            subprocess.check_output(
                [wrapper_script, image_path, model_param, "generate", analysis_type], 
                stderr=subprocess.STDOUT
            )
            
            # Find the output file generated by the wrapper
            if use_phi4:
                expected_output = f"{os.path.splitext(image_path)[0]}_enhanced_phi4.jpg"
            else:
                expected_output = f"{os.path.splitext(image_path)[0]}_enhanced.jpg"
                
            if os.path.exists(expected_output):
                # Copy to our output path
                with open(expected_output, "rb") as src_file:
                    with open(output_path, "wb") as dest_file:
                        dest_file.write(src_file.read())
                os.remove(expected_output)
        else:
            # Original approach: Choose the appropriate script
            script_path = os.path.join("llava", "demo", "gen_infer_phi4.sh" if use_phi4 else "gen_infer.sh")
            
            # Run the script with the analysis type
            subprocess.check_output(
                [script_path, image_path, output_path, analysis_type], 
                stderr=subprocess.STDOUT
            )
        
        # Read the output image
        if os.path.exists(output_path):
            with open(output_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            result = {
                'result_image': img_data,
                'model': "Phi-4" if use_phi4 else "Phi-3-mini",
                'analysis_type': analysis_type,
                'prompt': question,
                'detected': analysis_type != data.get('analysis_type') if 'analysis_type' in data else True  # Indicate if this was auto-detected
            }
        else:
            result = {'error': 'Failed to generate output image'}
    except subprocess.CalledProcessError as e:
        output = e.output.decode('utf-8') if isinstance(e.output, bytes) else str(e.output)
        result = {'error': f"Process error: {output}"}
    except Exception as e:
        result = {'error': f"Error processing request: {str(e)}"}
    finally:
        # Clean up
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(output_path):
            os.remove(output_path)
    
    return jsonify(result)

@app.route('/analysis-types', methods=['GET'])
def get_analysis_types():
    """Return available analysis types"""
    task = request.args.get('task', 'all')
    
    if task == 'all':
        return jsonify(ANALYSIS_TYPES)
    elif task in ANALYSIS_TYPES:
        return jsonify({task: ANALYSIS_TYPES[task]})
    else:
        return jsonify({"error": "Invalid task type"}), 400

@app.route('/models', methods=['GET'])
def get_models():
    """Return available models"""
    models = [
        {
            "id": "phi3",
            "name": "HealthGPT-M3 (Phi-3-mini)",
            "description": "A smaller version optimized for speed and reduced memory usage."
        }
    ]
    
    # Check if Phi-4 is available
    if os.path.exists(os.path.join("models", "phi-4")) and os.path.exists(os.path.join("weights", "com_hlora_weights_phi4.bin")):
        models.append({
            "id": "phi4",
            "name": "HealthGPT-L14 (Phi-4)",
            "description": "A larger version designed for higher performance and more complex tasks."
        })
    
    return jsonify({"models": models})

if __name__ == '__main__':
    print(f"Starting HealthGPT API on port 5001 with temp directory: {temp_dir}")
    app.run(host='0.0.0.0', port=5001)