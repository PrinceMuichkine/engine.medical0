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
import requests
from urllib.parse import urlparse
import time

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

def download_image_from_url(url):
    """
    Download an image from a URL and save it to a temporary file
    Returns the path to the downloaded file
    """
    try:
        logger.info(f"Downloading image from URL: {url}")
        
        # Parse the URL to get the filename
        parsed_url = urlparse(url)
        path = parsed_url.path
        filename = os.path.basename(path)
        
        # If no filename is found, generate a UUID
        if not filename or '.' not in filename:
            filename = f"download_{uuid.uuid4()}.jpg"
        
        # Create a temporary file path
        temp_path = os.path.join(temp_dir, filename)
        
        # Download the image with retry logic
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Use a session for better connection management
                with requests.Session() as session:
                    session.headers.update({
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    })
                    
                    # Download with timeout
                    response = session.get(url, stream=True, timeout=15)
                    response.raise_for_status()  # Raise an exception for error status codes
                    
                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('image/'):
                        logger.warning(f"Content type is not an image: {content_type}")
                        # Continue anyway, some servers don't set correct content type
                    
                    # Save the image to the temporary file
                    with open(temp_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Verify the file is a valid image
                    try:
                        img = Image.open(temp_path)
                        img.verify()  # Verify it's an image
                        logger.info(f"Successfully downloaded valid image to {temp_path}")
                        return temp_path
                    except Exception as img_err:
                        logger.error(f"Downloaded file is not a valid image: {str(img_err)}")
                        raise ValueError(f"Downloaded file is not a valid image: {str(img_err)}")
            
            except (requests.RequestException, ValueError) as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to download after {max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Retry {retry_count}/{max_retries} after error: {str(e)}")
                time.sleep(1)  # Wait before retrying
                
    except Exception as e:
        logger.error(f"Error downloading image from URL: {str(e)}", exc_info=True)
        raise

def find_weight_files():
    """Search for weight files in common locations and return full paths"""
    weight_files = {}
    
    # Common places to look
    search_dirs = [
        os.path.abspath('./weights'),  # Current directory weights
        os.path.abspath('../weights'),  # Parent directory weights
        os.path.abspath('../../weights'),  # Grandparent directory weights
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights'),  # Script directory weights
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights'),  # Parent of script directory
    ]
    
    # Files to look for
    files_to_find = [
        "com_hlora_weights.bin",
        "gen_hlora_weights.bin",
        "fusion_layer_weights.bin",
        "com_hlora_weights_phi4.bin"
    ]
    
    # Find the root weights directory that contains the most files
    root_weights_dir = None
    max_files_found = 0
    
    # Search each directory
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            logger.info(f"Checking for weight files in: {search_dir}")
            files_found = 0
            for file_name in files_to_find:
                file_path = os.path.join(search_dir, file_name)
                if os.path.exists(file_path):
                    weight_files[file_name] = file_path
                    logger.info(f"Found weight file: {file_name} at {file_path}")
                    files_found += 1
            
            # Keep track of directory with most weight files
            if files_found > max_files_found:
                max_files_found = files_found
                root_weights_dir = search_dir
    
    # If we found a directory with weights, ensure it's accessible from the script directory
    if root_weights_dir and max_files_found > 0:
        logger.info(f"Found primary weights directory at {root_weights_dir} with {max_files_found} weight files")
        
        # Create symlinks in the script directory (llava/demo) to make sure scripts can find the weights
        script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llava", "demo")
        if os.path.exists(script_dir):
            # Create a weights symlink in the script directory
            script_weights_dir = os.path.join(script_dir, "weights")
            if not os.path.exists(script_weights_dir):
                try:
                    logger.info(f"Creating symlink from {root_weights_dir} to {script_weights_dir}")
                    os.symlink(root_weights_dir, script_weights_dir)
                except Exception as e:
                    logger.warning(f"Error creating symlink in script directory: {str(e)}")
            else:
                logger.info(f"Script directory weights symlink already exists at {script_weights_dir}")
            
            # Also create a symlink at the ../../weights location referenced by the script
            parent_weights_dir = os.path.join(os.path.dirname(script_dir), "..", "weights")
            parent_weights_dir = os.path.normpath(parent_weights_dir)
            if not os.path.exists(parent_weights_dir):
                try:
                    # Create parent directory if needed
                    parent_dir = os.path.dirname(parent_weights_dir)
                    if not os.path.exists(parent_dir):
                        os.makedirs(parent_dir, exist_ok=True)
                    
                    logger.info(f"Creating parent symlink from {root_weights_dir} to {parent_weights_dir}")
                    os.symlink(root_weights_dir, parent_weights_dir)
                except Exception as e:
                    logger.warning(f"Error creating parent symlink: {str(e)}")
            else:
                logger.info(f"Parent directory weights symlink already exists at {parent_weights_dir}")
    
    if not weight_files:
        logger.error("No weight files found in any of the search directories")
    
    return weight_files

def _clean_extracted_answer(text):
    """Clean up extracted model answers to improve readability"""
    if not text:
        return text
    
    # Remove common debug markers and annotations
    clean_text = text
    
    # Remove script debug markers
    debug_patterns = [
        "Debug:", "Running with PID", "Script returned status code", "======", "-----",
        "End Response", "Response:", "HealthGPT Response", "HealthGPT:", "Output:"
    ]
    
    for pattern in debug_patterns:
        if pattern in clean_text:
            parts = clean_text.split(pattern)
            # Keep the most substantial part
            substantial_part = max(parts, key=lambda x: len(x.strip()))
            clean_text = substantial_part.strip()
    
    # Remove extra newlines
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    
    # Remove any remaining debug lines
    lines = clean_text.split('\n')
    content_lines = []
    for line in lines:
        if not any(marker in line for marker in ["DEBUG:", "INFO:", "WARNING:", "ERROR:", "====", "----"]):
            content_lines.append(line)
    
    clean_text = '\n'.join(content_lines)
    
    return clean_text.strip()

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check endpoint accessed")
    return jsonify({"status": "ok", "message": "HealthGPT API is running"}), 200

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """Endpoint for medical image analysis"""
    logger.info("Analyze image endpoint accessed")
    data = request.json
    if not data:
        logger.error("Missing JSON data in request")
        return jsonify({"error": "Missing request data"}), 400
    
    # Optional parameters
    analysis_type = data.get('analysis_type')  # Optional
    question = data.get('question')  # Optional
    use_phi4 = data.get('use_phi4', False)  # Default to Phi-3 unless specified
    weights_path = data.get('weights_path')  # Optional custom weights path
    base_path = data.get('base_path')  # Optional base path
    debug = data.get('debug', False)  # Debug mode

    # Find available weight files before running the script
    weight_files = find_weight_files()
    
    # If weights_path is provided, ensure it exists
    if weights_path:
        logger.info(f"Custom weights path provided: {weights_path}")
        abs_weights_path = os.path.abspath(weights_path)
        if os.path.exists(abs_weights_path):
            logger.info(f"Weights path exists: {abs_weights_path}")
            # Check if the H-LoRA weights file exists
            hlora_weights_path = os.path.join(abs_weights_path, "com_hlora_weights.bin")
            if os.path.exists(hlora_weights_path):
                logger.info(f"H-LoRA weights file exists: {hlora_weights_path}")
                # Create symlink in case the script expects files in a different location
                try:
                    # Create symlink in llava/demo if it doesn't exist
                    script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llava", "demo")
                    target_dir = os.path.join(script_dir, "weights")
                    if not os.path.exists(target_dir):
                        logger.info(f"Creating symlink from {abs_weights_path} to {target_dir}")
                        os.symlink(abs_weights_path, target_dir)
                except Exception as e:
                    logger.warning(f"Error creating symlink: {str(e)}")
            else:
                logger.error(f"H-LoRA weights file not found at {hlora_weights_path}")
        else:
            logger.error(f"Provided weights path does not exist: {abs_weights_path}")

    # Ensure the script can find the weights in the location it expects (../../weights)
    # Get script directory
    script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llava", "demo")
    
    # Direct relative path correction - create the directory structure the script expects
    expected_weights_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'weights'))
    
    # Ensure the expected directory exists
    os.makedirs(os.path.dirname(expected_weights_dir), exist_ok=True)
    
    # Create symlink if it doesn't exist and we have weights elsewhere
    if not os.path.exists(expected_weights_dir) and weight_files:
        # Find the actual weights directory
        source_weights_dir = os.path.dirname(next(iter(weight_files.values())))
        logger.info(f"Creating symlink from {source_weights_dir} to {expected_weights_dir}")
        try:
            os.symlink(source_weights_dir, expected_weights_dir)
        except Exception as e:
            logger.warning(f"Error creating expected weights symlink: {str(e)}")
            
            # If symlink fails, try direct copying of weight files
            try:
                logger.info(f"Attempting to copy weight files directly to {expected_weights_dir}")
                os.makedirs(expected_weights_dir, exist_ok=True)
                for file_name, file_path in weight_files.items():
                    target_path = os.path.join(expected_weights_dir, file_name)
                    if not os.path.exists(target_path):
                        logger.info(f"Copying {file_path} to {target_path}")
                        import shutil
                        shutil.copy2(file_path, target_path)
            except Exception as copy_err:
                logger.error(f"Error copying weight files: {str(copy_err)}")

    logger.info(f"Analyze request details: analysis_type={analysis_type}, use_phi4={use_phi4}, question_length={len(question) if question else 0}, weights_path={weights_path}, debug={debug}")
    
    image_path = None
    downloaded_path = None  # Track if we downloaded a remote image
    
    try:
        # Handle different image input formats
        if 'image' in data:
            # First, check if it's a remote URL (simplest case)
            if isinstance(data['image'], str) and (data['image'].startswith('http://') or data['image'].startswith('https://')):
                logger.info(f"Detected remote image URL: {data['image'][:100]}...")
                try:
                    image_path = download_image_from_url(data['image'])
                    downloaded_path = image_path  # Remember that we downloaded this
                    logger.info(f"Successfully downloaded remote image to {image_path}")
                except Exception as e:
                    logger.error(f"Failed to download remote image: {str(e)}")
                    return jsonify({"error": f"Failed to download image from URL: {str(e)}"}), 400
            
            # Then check if it's a data URL
            elif isinstance(data['image'], str) and data['image'].startswith('data:'):
                # Decode base64 image
                logger.info("Processing base64 image data")
                try:
                    image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
                    image_path = os.path.join(temp_dir, f"upload_{uuid.uuid4()}.jpg")
                    
                    logger.info(f"Saving uploaded image to {image_path}")
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                except Exception as e:
                    logger.error(f"Failed to decode base64 image: {str(e)}")
                    return jsonify({"error": f"Invalid base64 image data: {str(e)}"}), 400
            else:
                logger.error("Image data format not recognized")
                return jsonify({"error": "Image data format not recognized. Please provide a valid URL or base64 encoded image."}), 400
        else:
            logger.error("Missing image data in request")
            return jsonify({"error": "Missing image data"}), 400
        
        # Auto-detect analysis type if not provided
        if not analysis_type:
            try:
                with open(image_path, "rb") as f:
                    image_data = f.read()
                analysis_type = detect_analysis_type(image_data, question, "analyze")
                logger.info(f"Auto-detected analysis type: {analysis_type}")
            except Exception as e:
                logger.error(f"Error during analysis type detection: {str(e)}")
                analysis_type = "general"  # Fallback to general
        
        # If no question is provided, try to get one from the medical0.tools.txt file
        if not question:
            # Try to get question from tools file first
            question = extract_prompt_from_tools_file(analysis_type, "analyze")
            
            # Default fallback if not found
            if not question:
                question = "Analyze this medical image. Identify the imaging modality, describe visible anatomical structures, and note any abnormalities, congenital variations, or developmental anomalies. Include observations about organ position, shape, and symmetry."
            
            logger.info(f"Using default question for analysis: {question[:50]}...")
        
        # Use our wrapper script if it exists, otherwise fall back to original approach
        wrapper_script = os.path.join("llava", "demo", "medical0_analyze.sh")
        if os.path.exists(wrapper_script):
            # Use the wrapper script with all parameters
            logger.info(f"Using wrapper script: {wrapper_script}")
            model_param = "phi4" if use_phi4 else "phi3"
            
            # Build environment variables to help locate weights
            env = os.environ.copy()
            if weight_files:
                # Add weight file paths as environment variables
                for file_name, file_path in weight_files.items():
                    env[f"HLORA_{file_name.replace('.', '_').upper()}"] = file_path
                    logger.info(f"Set environment variable HLORA_{file_name.replace('.', '_').upper()}={file_path}")
            
            # Add debugging information
            if debug:
                logger.info(f"Current working directory: {os.getcwd()}")
                logger.info(f"Script absolute path: {os.path.abspath(wrapper_script)}")
                for file_name in ["com_hlora_weights.bin", "gen_hlora_weights.bin"]:
                    # Check a few typical locations
                    for path in ["./weights", "../weights", "../../weights", "weights"]:
                        test_path = os.path.join(os.path.abspath(path), file_name)
                        logger.info(f"Testing path {test_path}: exists={os.path.exists(test_path)}")
            
            # Run the command with environment variables
            output = subprocess.check_output(
                [wrapper_script, image_path, model_param, "analyze", analysis_type], 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env
            )
        else:
            # Original approach: Choose the appropriate script
            script_path = os.path.join("llava", "demo", "com_infer_phi4.sh" if use_phi4 else "com_infer.sh")
            logger.info(f"Using script: {script_path}")
            
            # Run the script with the analysis type
            output = subprocess.check_output(
                [script_path, image_path, analysis_type], 
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
        
        # Log the raw output for debugging
        logger.info(f"Raw script output (first 500 chars): {output[:500]}")
        
        # Check if the script execution actually failed (the subprocess didn't raise an exception,
        # but the script itself reported failure in its output)
        if "Inference failed with exit code" in output or "CPU inference also failed" in output:
            logger.error("Model inference failed according to script output")
            return jsonify({
                "error": "Model inference failed. The model was unable to process the image successfully.",
                "details": "The inference engine reported a failure. This could be due to GPU issues, image format problems, or missing model weights."
            }), 500
        
        # Enhanced answer extraction - try multiple patterns
        answer = None
        
        # Look for the response between model's output
        pattern = r"<assistant>[\s\S]*?</assistant>"
        if re.search(pattern, output, re.IGNORECASE):
            match = re.search(pattern, output, re.IGNORECASE)
            answer = match.group(0)
            # Remove the tags
            answer = answer.replace("<assistant>", "").replace("</assistant>", "").strip()
            logger.info(f"Extracted answer using assistant tags (length: {len(answer)})")
        
        # Try standard format first: "HealthGPT: [answer]"
        elif "HealthGPT: " in output:
            answer = output.split("HealthGPT: ")[1].strip()
            # Further clean up - remove any trailing sections after "====="
            if "=======" in answer:
                answer = answer.split("=======")[0].strip()
            logger.info(f"Successfully extracted answer using 'HealthGPT:' marker (length: {len(answer)})")
        
        # Try to extract from Python's print output "answer = [answer]"
        elif "answer = " in output:
            try:
                answer = output.split("answer = ")[1].split("\n")[0].strip()
                logger.info(f"Extracted answer using 'answer =' pattern (length: {len(answer)})")
                
                # Strip quotes if present
                if (answer.startswith('"') and answer.endswith('"')) or (answer.startswith("'") and answer.endswith("'")):
                    answer = answer[1:-1]
            except:
                logger.warning("Failed to extract answer using 'answer =' pattern")
        
        # Try to find any response between triple backticks which often indicates the response
        elif "```" in output:
            try:
                parts = output.split("```")
                if len(parts) >= 3:  # At least one block between backticks
                    answer = parts[1].strip()
                    logger.info(f"Extracted answer from code block (length: {len(answer)})")
            except:
                logger.warning("Failed to extract answer from code block")
        
        # Look for "Response:" or "Output:" markers
        elif "Response:" in output or "Output:" in output:
            try:
                if "Response:" in output:
                    answer = output.split("Response:")[1].strip()
                else:
                    answer = output.split("Output:")[1].strip()
                
                # Trim if there are more markers
                if "\n" in answer:
                    # Get lines until the next seemingly header-like line (ending with :)
                    lines = []
                    for line in answer.split("\n"):
                        if line.strip().endswith(":") and len(line.strip()) < 30:
                            break
                        lines.append(line)
                    answer = "\n".join(lines).strip()
                
                logger.info(f"Extracted answer using Response/Output marker (length: {len(answer)})")
            except:
                logger.warning("Failed to extract answer using Response/Output marker")
                
        # As a last resort, take the last part after "=======" line (common in outputs)
        elif "=======" in output:
            try:
                sections = output.split("=======")
                # Find the most meaningful section (largest non-debug section)
                meaningful_content = ""
                for section in sections:
                    # Skip debug/status lines
                    if not any(x in section.lower() for x in ["debug:", "error:", "status", "code:", "pid", "exiting", "ealthgpt response", "nd response"]):
                        # If it has substantial text content
                        if len(section.strip()) > 100:
                            if len(section.strip()) > len(meaningful_content):
                                meaningful_content = section.strip()
                
                if meaningful_content:
                    answer = meaningful_content
                    logger.info(f"Extracted answer from meaningful section (length: {len(answer)})")
                else:
                    # Default to last section if nothing better is found
                    answer = sections[-1].strip()
                    logger.info(f"Extracted answer from last section (length: {len(answer)})")
            except:
                logger.warning("Failed to extract answer from sections")
        
        # Fallback if no extraction method worked
        if not answer:
            # Log the entire output to help diagnose what's wrong
            logger.warning(f"Could not extract answer from output. Full output: {output}")
            
            # Try to extract any reasonable text content
            lines = output.strip().split("\n")
            # Find lines that are likely meaningful content (longer than 50 chars and not debug)
            content_lines = []
            for line in lines:
                line = line.strip()
                if len(line) > 50 and not line.startswith(("#", "DEBUG", "INFO", "WARNING", "ERROR", "===", "---")):
                    # Skip lines that are part of common output patterns but not valuable content
                    if not any(x in line.lower() for x in ["activating conda", "started", "running with pid", "importing", "loading", "loaded", "cuda", "gpu", "device", "tensor", "weight"]):
                        content_lines.append(line)
            
            if content_lines:
                answer = "\n".join(content_lines)
                logger.info(f"Using content lines as fallback (length: {len(answer)})")
                
                # Additional validation - if we're using fallback content, check if it's just script output
                # rather than actual model output
                if "HealthGPT-M3" in answer or "weights_path" in answer or "inference" in answer.lower():
                    logger.error("Fallback content appears to be script output, not model analysis")
                    return jsonify({
                        "error": "The model failed to generate a proper analysis",
                        "details": "The script executed, but the model did not produce meaningful output. This may indicate an issue with model weights or GPU configuration."
                    }), 500
            else:
                # Final fallback - return a helpful error message
                answer = "The model processed your image but didn't generate a meaningful response. This could be due to an issue with the image format, quality, or model configuration. Please try a different image or contact support."
                logger.warning("No meaningful content found in output")
        
        # Clean up the answer for better presentation
        answer = _clean_extracted_answer(answer)
        
        result = {
            "answer": answer,
            "model": "Phi-4" if use_phi4 else "Phi-3-mini",
            "analysis_type": analysis_type,
            "prompt": question,
            "detected": analysis_type != data.get('analysis_type') if 'analysis_type' in data else True  # Indicate if this was auto-detected
        }
        
        logger.info(f"Returning successful analysis result for {analysis_type}")
        
    except subprocess.CalledProcessError as e:
        error_output = e.output
        logger.error(f"Process error in analysis: {error_output}")
        result = {"error": f"Process error: {error_output}"}
    except Exception as e:
        logger.error(f"Error processing analysis request: {str(e)}", exc_info=True)
        result = {"error": f"Error processing request: {str(e)}"}
    finally:
        # Clean up
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
            logger.debug(f"Removed temporary file: {image_path}")
    
    return jsonify(result)

@app.route('/reconstruct-image', methods=['POST'])
def reconstruct_image():
    """Endpoint for image reconstruction"""
    logger.info("Reconstruct image endpoint accessed")
    data = request.json
    if not data:
        logger.error("Missing JSON data in request")
        return jsonify({"error": "Missing request data"}), 400
    
    # Optional parameters
    analysis_type = data.get('analysis_type')  # Optional
    question = data.get('question')  # Optional
    use_phi4 = data.get('use_phi4', False)  # Default to Phi-3 unless specified
    
    logger.info(f"Reconstruction request details: analysis_type={analysis_type}, use_phi4={use_phi4}, question_length={len(question) if question else 0}")
    
    image_path = None
    output_path = None
    downloaded_path = None  # Track if we downloaded a remote image
    
    try:
        # Handle different image input formats
        if 'image' in data:
            # Check if it's a data URL
            if isinstance(data['image'], str) and data['image'].startswith('data:'):
                # Decode base64 image
                logger.info("Processing base64 image data")
                image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
                image_path = os.path.join(temp_dir, f"upload_{uuid.uuid4()}.jpg")
                output_path = os.path.join(temp_dir, f"output_{uuid.uuid4()}.jpg")
                
                logger.info(f"Saving uploaded image to {image_path}")
                with open(image_path, "wb") as f:
                    f.write(image_data)
            
            # Check if it's a remote URL
            elif isinstance(data['image'], str) and (data['image'].startswith('http://') or data['image'].startswith('https://')):
                logger.info(f"Detected remote image URL: {data['image'][:100]}...")
                try:
                    image_path = download_image_from_url(data['image'])
                    downloaded_path = image_path  # Remember that we downloaded this
                    output_path = os.path.join(temp_dir, f"output_{uuid.uuid4()}.jpg")
                    logger.info(f"Successfully downloaded remote image to {image_path}")
                except Exception as e:
                    logger.error(f"Failed to download remote image: {str(e)}")
                    return jsonify({"error": f"Failed to download image from URL: {str(e)}"}), 400
            else:
                logger.error("Image data format not recognized")
                return jsonify({"error": "Image data format not recognized"}), 400
        else:
            logger.error("Missing image data in request")
            return jsonify({"error": "Missing image data"}), 400
        
        # Auto-detect analysis type if not provided
        if not analysis_type:
            try:
                with open(image_path, "rb") as f:
                    image_data = f.read()
                analysis_type = detect_analysis_type(image_data, question, "generate")
                logger.info(f"Auto-detected generation type: {analysis_type}")
            except Exception as e:
                logger.error(f"Error during analysis type detection: {str(e)}")
                analysis_type = "general"  # Fallback to general
        
        # If no question is provided, try to get one from the medical0.tools.txt file
        if not question:
            # Try to get question from tools file first
            question = extract_prompt_from_tools_file(analysis_type, "generate")
            
            # Default fallback if not found
            if not question:
                question = "Reconstruct this medical image with enhanced clarity. Highlight any anatomical abnormalities, congenital variations, or pathological findings. Pay special attention to organ position, structural relationships, and any asymmetries."
            
            logger.info(f"Using default question for reconstruction: {question[:50]}...")
        
        # Use our wrapper script if it exists, otherwise fall back to original approach
        wrapper_script = os.path.join("llava", "demo", "medical0_analyze.sh")
        if os.path.exists(wrapper_script):
            # Use the wrapper script with all parameters
            logger.info(f"Using wrapper script: {wrapper_script}")
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
                logger.info(f"Found expected output file: {expected_output}")
                # Copy to our output path
                with open(expected_output, "rb") as src_file:
                    with open(output_path, "wb") as dest_file:
                        dest_file.write(src_file.read())
                os.remove(expected_output)
        else:
            # Original approach: Choose the appropriate script
            script_path = os.path.join("llava", "demo", "gen_infer_phi4.sh" if use_phi4 else "gen_infer.sh")
            logger.info(f"Using script: {script_path}")
            
            # Run the script with the analysis type
            output = subprocess.check_output(
                [script_path, image_path, output_path, analysis_type], 
                stderr=subprocess.STDOUT
            )
        
        # Read the output image
        if os.path.exists(output_path):
            logger.info(f"Reading generated output image from {output_path}")
            with open(output_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            result = {
                'result_image': img_data,
                'model': "Phi-4" if use_phi4 else "Phi-3-mini",
                'analysis_type': analysis_type,
                'prompt': question,
                'detected': analysis_type != data.get('analysis_type') if 'analysis_type' in data else True  # Indicate if this was auto-detected
            }
            logger.info("Successfully returned reconstructed image")
        else:
            logger.error(f"Output image file not found: {output_path}")
            result = {'error': 'Failed to generate output image'}
    except subprocess.CalledProcessError as e:
        output = e.output.decode('utf-8') if isinstance(e.output, bytes) else str(e.output)
        logger.error(f"Process error in reconstruction: {output}")
        result = {'error': f"Process error: {output}"}
    except Exception as e:
        logger.error(f"Error processing reconstruction request: {str(e)}", exc_info=True)
        result = {'error': f"Error processing request: {str(e)}"}
    finally:
        # Clean up
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
            logger.debug(f"Removed temporary file: {image_path}")
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
            logger.debug(f"Removed temporary file: {output_path}")
    
    return jsonify(result)

@app.route('/analysis-types', methods=['GET'])
def get_analysis_types():
    """Return available analysis types"""
    task = request.args.get('task', 'all')
    logger.info(f"Analysis types endpoint accessed for task: {task}")
    
    if task == 'all':
        return jsonify(ANALYSIS_TYPES)
    elif task in ANALYSIS_TYPES:
        return jsonify({task: ANALYSIS_TYPES[task]})
    else:
        logger.warning(f"Invalid task type requested: {task}")
        return jsonify({"error": "Invalid task type"}), 400

@app.route('/models', methods=['GET'])
def get_models():
    """Return available models"""
    logger.info("Models endpoint accessed")
    
    # Always return both models regardless of file availability
    # This ensures the frontend can always show both options
    models = [
        {
            "id": "phi3",
            "name": "HealthGPT-M3 (Phi-3-mini)",
            "description": "A smaller version optimized for speed and reduced memory usage."
        },
        {
            "id": "phi4",
            "name": "HealthGPT-L14 (Phi-4)",
            "description": "A larger version designed for higher performance and more complex tasks."
        }
    ]
    
    # Check for actual file availability but return both models regardless
    phi3_available = True  # We know Phi-3 is working
    phi4_available = os.path.exists(os.path.join("weights", "com_hlora_weights_phi4.bin"))
    
    logger.info(f"Model availability: Phi-3={phi3_available}, Phi-4={phi4_available}")
    
    # Add availability flag to response
    for model in models:
        if model["id"] == "phi3":
            model["available"] = phi3_available
        elif model["id"] == "phi4":
            model["available"] = phi4_available
    
    return jsonify({"models": models})

if __name__ == '__main__':
    print(f"Starting HealthGPT API on port 5001 with temp directory: {temp_dir}")
    
    # Initialize weight path resolution at startup
    try:
        print("Setting up weight file paths...")
        weight_files = find_weight_files()
        
        if weight_files:
            # Create symlinks to the weights directory in common locations
            # This ensures scripts can find the weights regardless of their working directory
            source_dir = os.path.dirname(next(iter(weight_files.values())))
            print(f"Found weights directory: {source_dir}")
            
            # Common locations where scripts expect to find weights
            target_locations = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "llava", "demo", "weights"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights"),
                os.path.join(os.path.abspath("."), "weights"),
            ]
            
            for target in target_locations:
                try:
                    target_dir = os.path.dirname(target)
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir, exist_ok=True)
                    
                    if not os.path.exists(target):
                        print(f"Creating symlink: {source_dir} -> {target}")
                        os.symlink(source_dir, target)
                    else:
                        print(f"Symlink target already exists: {target}")
                except Exception as e:
                    print(f"Warning: Could not create symlink to {target}: {str(e)}")
            
            print("Weight path initialization complete")
        else:
            print("WARNING: No weight files found during initialization!")
    except Exception as e:
        print(f"Error during weight path initialization: {str(e)}")
    
    app.run(host='0.0.0.0', port=5001)