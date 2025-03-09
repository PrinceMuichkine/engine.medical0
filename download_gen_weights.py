from huggingface_hub import hf_hub_download
import os

os.makedirs("weights", exist_ok=True)

print("Downloading HealthGPT-L14 generation weights...")
try:
    # Download the specific weights file
    file_path = hf_hub_download(
        repo_id="lintw/HealthGPT-L14", 
        filename="gen_hlora_weights_phi4.bin",
        local_dir="weights",
        local_dir_use_symlinks=False
    )
    print(f"Successfully downloaded weights to {file_path}")
except Exception as e:
    print(f"Error downloading weights: {e}")
    print("Note: gen_hlora_weights_phi4.bin might not be available yet. We'll use the Phi-3 weights for generation.")