import os
import base64
import json
import io
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image."""
    if base64_string.startswith('data:image'):
        # Remove the data URL scheme if present
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def generate_mock_medical_image(text, width=512, height=512):
    """Generate a simple mock medical image with the prompt text."""
    # Create a simple mock image
    image = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    
    # Add a simple mock medical structure
    draw.ellipse((100, 100, 400, 400), fill=(200, 200, 200), outline=(100, 100, 100))
    draw.line((100, 250, 400, 250), fill=(100, 100, 100), width=2)
    draw.line((250, 100, 250, 400), fill=(100, 100, 100), width=2)
    
    # Add text
    try:
        # Try to load a font, use default if not available
        font = ImageFont.truetype("Arial", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Add the prompt text to the image, wrapping if necessary
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + word + " "
        # Estimate width to keep lines reasonable
        if len(test_line) > 40:  # Roughly 40 chars per line
            lines.append(current_line)
            current_line = word + " "
        else:
            current_line = test_line
    lines.append(current_line)  # Add the last line
    
    # Draw the text
    y_position = 420
    for line in lines:
        draw.text((50, y_position), line, fill=(0, 0, 0), font=font)
        y_position += 25
    
    # Add a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    draw.text((10, 10), f"Mock HealthGPT - {timestamp}", fill=(50, 50, 50), font=font)
    
    return image

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint to check if API is running."""
    return jsonify({
        "status": "healthy",
        "message": "HealthGPT API (MOCK) is running",
        "models": ["HealthGPT-M3-COM", "HealthGPT-M3-GEN", "HealthGPT-L14-COM"]
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Mock endpoint to analyze medical images."""
    try:
        data = request.json
        question = data.get('question', '')
        image_base64 = data.get('image', None)
        model = data.get('model', 'HealthGPT-M3')
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
            
        if not image_base64:
            return jsonify({"error": "Image is required"}), 400

        # Generate a mock analysis response
        mock_response = f"[MOCK HealthGPT {model}] Analysis of the medical image:\n\n"
        
        if "x-ray" in question.lower():
            mock_response += "This appears to be an X-ray image. I can see what looks like bone structures.\n\n"
        elif "mri" in question.lower():
            mock_response += "This appears to be an MRI scan. I can observe soft tissue contrast.\n\n"
        elif "ct" in question.lower():
            mock_response += "This appears to be a CT scan showing cross-sectional images.\n\n"
        else:
            mock_response += "This appears to be a medical image. I can see various anatomical structures.\n\n"
        
        mock_response += f"Regarding your question: '{question}'\n\n"
        mock_response += "Based on the image, I can observe normal anatomical structures without obvious abnormalities. "
        mock_response += "However, please note that this is a mock response and not an actual medical diagnosis. "
        mock_response += "Always consult with a qualified healthcare professional for proper medical advice and diagnosis."
        
        return jsonify({
            "answer": mock_response,
            "type": "text"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_image():
    """Mock endpoint to generate medical images based on text descriptions."""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        reference_image_base64 = data.get('referenceImage', None)
        model = data.get('model', 'HealthGPT-M3')
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Check if model supports generation
        if model.upper() == "HEALTHGPT-L14":
            return jsonify({"error": "HealthGPT-L14 does not support image generation"}), 400
        
        # Generate a mock medical image based on the prompt
        mock_image = generate_mock_medical_image(prompt)
        
        # Convert the generated image to base64
        buffered = io.BytesIO()
        mock_image.save(buffered, format="PNG")
        mock_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            "image": mock_image_base64,
            "type": "image"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Endpoint to list available models and their capabilities."""
    return jsonify({
        "models": [
            {
                "id": "HealthGPT-M3",
                "name": "HealthGPT M3 (Mock)",
                "capabilities": ["analyze", "generate"],
                "description": "Mock medical vision-language model based on Phi-3-mini"
            },
            {
                "id": "HealthGPT-L14",
                "name": "HealthGPT L14 (Mock)",
                "capabilities": ["analyze"],
                "description": "Mock advanced medical vision-language model based on Phi-4"
            }
        ]
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Mock endpoint for text-based medical questions."""
    try:
        data = request.json
        message = data.get('message', '')
        model = data.get('model', 'HealthGPT-M3')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Generate a mock medical response
        mock_response = f"[MOCK HealthGPT {model}] Medical response:\n\n"
        
        if "symptoms" in message.lower() or "diagnosis" in message.lower():
            mock_response += "Based on the symptoms you've described, there could be several potential causes. "
            mock_response += "However, it's important to consult with a healthcare professional for proper diagnosis. "
            mock_response += "This is a simulated response for demonstration purposes only."
        elif "treatment" in message.lower() or "medication" in message.lower():
            mock_response += "Treatment options typically depend on the specific condition and patient history. "
            mock_response += "This mock response cannot provide specific medical advice. "
            mock_response += "Please consult with a qualified healthcare provider for appropriate treatment recommendations."
        else:
            mock_response += "Thank you for your medical question. This is a mock AI response for demonstration purposes. "
            mock_response += "In a real medical AI system, I would provide relevant information based on current medical knowledge. "
            mock_response += "However, AI systems should complement, not replace, the advice of healthcare professionals."
        
        return jsonify({
            "answer": mock_response,
            "type": "text"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses."""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    # Use port 8000 instead of 5000 to avoid conflict with AirPlay
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting Mock HealthGPT API on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True) 