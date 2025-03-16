import argparse
import json
import time
import requests
import os

def get_prompt_from_type(prompt_type):
    """Get a specialized prompt based on prompt type, similar to com_infer.sh"""
    prompt_file = "../../medical0.tools.txt"
    
    # Default comprehensive analysis prompt
    default_prompt = "Analyze this medical image. Identify the imaging modality, describe visible anatomical structures, and note any abnormalities, congenital variations, or developmental anomalies. Include observations about organ position, shape, and symmetry."
    
    if not os.path.isfile(prompt_file) or prompt_type == "general":
        return default_prompt
        
    try:
        prompt_mapping = {
            "modality": "a) Modality Recognition:",
            "anatomy": "b) Anatomical Mapping:",
            "abnormality": "c) Abnormality Detection:",
            "congenital": "d) Congenital Variant Recognition:",
            "thoracic": "a) Thoracic Analysis:",
            "abdominal": "b) Abdominal Assessment:",
            "neuro": "c) Neuroimaging Interpretation:",
            "brain_viability": "f) Brain Viability Assessment:",
            "msk": "d) Musculoskeletal Examination:",
            "genitourinary": "e) Genitourinary System Analysis:",
            "diagnosis": "a) Differential Diagnosis:"
        }
        
        if prompt_type not in prompt_mapping:
            return default_prompt
            
        search_term = prompt_mapping[prompt_type]
        
        with open(prompt_file, 'r') as f:
            content = f.read()
            
        # Extract the prompt using similar logic to com_infer.sh
        # Find section with search term and extract the example prompt
        for line in content.split('\n'):
            if search_term in line:
                for next_line in content.split('\n'):
                    if "Example prompt:" in next_line:
                        # Extract text between quotes
                        start = next_line.find('"')
                        end = next_line.rfind('"')
                        if start != -1 and end != -1:
                            return next_line[start+1:end]
        
        return default_prompt
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return default_prompt

def analyze_image(server_url, img_path, question=None, prompt_type="general"):
    """Submit an image analysis request to the model server"""
    if question is None:
        question = get_prompt_from_type(prompt_type)
    
    url = f"{server_url}/api/analyze"
    payload = {
        "img_path": img_path,
        "question": question
    }
    
    print(f"Sending request to analyze: {img_path}")
    if prompt_type != "general":
        print(f"Using specialized prompt type: {prompt_type}")
    print(f"Question: {question}")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        total_time = time.time() - start_time
        
        print("\n" + "-" * 50)
        print(f"HealthGPT Analysis:")
        print("-" * 50)
        print(f"Q: {question}")
        print(f"HealthGPT: {result['response']}")
        print("-" * 50)
        print(f"Server processing time: {result['timing']['total_time']:.2f} seconds")
        print(f"Inference time: {result['timing']['inference_time']:.2f} seconds")
        print(f"Total round-trip time: {total_time:.2f} seconds")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Server response: {e.response.text}")
        return None

def check_server_status(server_url):
    """Check if the server is running and model is loaded"""
    try:
        response = requests.get(f"{server_url}/api/healthcheck")
        if response.status_code == 200:
            data = response.json()
            if data.get("model_loaded", False):
                print("Server is running and model is loaded")
                return True
            else:
                print("Server is running but model is not loaded yet")
                return False
        else:
            print(f"Server returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("Server is not running or not reachable")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Client for HealthGPT model server')
    parser.add_argument('--server_url', type=str, default='http://localhost:5000', 
                        help='URL of the model server')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to the image for analysis')
    parser.add_argument('--question', type=str, default=None,
                        help='Custom question to ask about the image')
    parser.add_argument('--prompt_type', type=str, default="general",
                        choices=["general", "modality", "anatomy", "abnormality", "congenital", 
                                "thoracic", "abdominal", "neuro", "brain_viability", "msk", 
                                "genitourinary", "diagnosis"],
                        help='Type of specialized prompt to use')
    
    args = parser.parse_args()
    
    # Check if server is ready
    if check_server_status(args.server_url):
        analyze_image(args.server_url, args.img_path, args.question, args.prompt_type)
    else:
        print("Please make sure the server is running before using this client.") 