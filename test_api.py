import requests
import base64
from PIL import Image
import io

# Path to your test image
image_path = "/workspace/healthgpt/image-test.png"

# Read the image and convert to base64
with open(image_path, "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

# API endpoint
url = "http://localhost:5000/api/analyze"

# Prepare payload
payload = {
    "image": image_data,
    "prompt": "What medical condition is shown in this image? Provide a detailed analysis."
}

# Make the request
response = requests.post(url, json=payload)

# Print the response
print(response.status_code)
print(response.json())