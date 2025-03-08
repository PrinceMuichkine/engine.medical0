from flask import Flask, request, jsonify
from flask_cors import CORS
from model import HealthGPT, HealthGPT_Agent
from config import HealthGPTConfig_M3_COM, HealthGPTConfig_M3_GEN, HealthGPTConfig_L14_COM
import base64
from io import BytesIO
from PIL import Image
import traceback
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js to communicate with this API

# Initialize HealthGPT agent
configs = {
    "HealthGPT-M3-COM": HealthGPTConfig_M3_COM(),
    "HealthGPT-M3-GEN": HealthGPTConfig_M3_GEN(),
    "HealthGPT-L14-COM": HealthGPTConfig_L14_COM()
}

agent = HealthGPT_Agent(configs=configs, model_name=None)

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image."""
    if base64_string.startswith('data:image'):
        # Remove the data URL scheme if present
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint to check if API is running."""
    return jsonify({"status": "healthy", "message": "HealthGPT API is running"})

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Endpoint to analyze medical images."""
    try:
        data = request.json
        question = data.get('question', '')
        image_base64 = data.get('image', None)
        model = data.get('model', 'HealthGPT-M3')
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
            
        if not image_base64:
            return jsonify({"error": "Image is required"}), 400

        # Convert base64 to PIL Image
        image = base64_to_image(image_base64)
        
        # Load model and analyze
        model_name = f"{model}-COM"
        agent.load_model(model_name=model_name)
        response = agent.process("Analyze Image", question, image)
        
        return jsonify({
            "answer": response,
            "type": "text"
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_image():
    """Endpoint to generate medical images based on text descriptions."""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        reference_image_base64 = data.get('referenceImage', None)
        model = data.get('model', 'HealthGPT-M3')
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Convert base64 to PIL Image if provided
        reference_image = None
        if reference_image_base64:
            reference_image = base64_to_image(reference_image_base64)
        
        # Check if model supports generation
        if model == "HealthGPT-L14":
            return jsonify({"error": "HealthGPT-L14 does not support image generation"}), 400
            
        # Load model and generate
        model_name = f"{model}-GEN"
        agent.load_model(model_name=model_name)
        generated_image = agent.process("Generate Image", prompt, reference_image)
        
        # Convert generated image to base64
        generated_image_base64 = image_to_base64(generated_image)
        
        return jsonify({
            "image": generated_image_base64,
            "type": "image"
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Endpoint to list available models."""
    return jsonify({
        "models": [
            {
                "id": "HealthGPT-M3",
                "name": "HealthGPT M3",
                "capabilities": ["analyze", "generate"],
                "description": "Medical vision-language model based on Phi-3-mini"
            },
            {
                "id": "HealthGPT-L14",
                "name": "HealthGPT L14",
                "capabilities": ["analyze"],
                "description": "Advanced medical vision-language model based on Phi-4"
            }
        ]
    })

if __name__ == '__main__':
    # Get port from environment variable or use default 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 