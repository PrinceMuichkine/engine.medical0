from huggingface_hub import login, snapshot_download
import os
from getpass import getpass

# Authenticate with explicit token input
print("Please get a token from https://huggingface.co/settings/tokens")
token = getpass("Enter your Hugging Face token: ")
login(token=token)  # This will store the token

print("Downloading the Phi-4 model...")
try:
    # Download the full model
    model_path = snapshot_download(
        repo_id="microsoft/phi-4",
        local_dir="models/phi-4", 
        local_dir_use_symlinks=False,
        ignore_patterns=["*.safetensors", "*.h5"],  # We want the PyTorch files
    )
    print(f"Successfully downloaded Phi-4 model to {model_path}")
except Exception as e:
    print(f"Error downloading model: {e}")
    print("Please check your token and internet connection.")