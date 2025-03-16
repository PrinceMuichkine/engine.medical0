#!/usr/bin/env python3
"""
Script to download required models from HuggingFace before starting the server.
This ensures that models are properly cached locally before attempting to load them.
"""

import os
import argparse
import logging
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('download_models')

def main():
    parser = argparse.ArgumentParser(description="Download models from HuggingFace")
    parser.add_argument("--cache-dir", type=str, default="hf_cache", 
                        help="Directory to cache models (default: hf_cache)")
    args = parser.parse_args()
    
    # Create cache directory
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables for HuggingFace
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    
    logger.info(f"Using cache directory: {cache_dir}")
    
    # List of models to download
    models = [
        "microsoft/Phi-3-mini-4k-instruct",
        "openai/clip-vit-large-patch14-336"
    ]
    
    for model_name in models:
        try:
            logger.info(f"Downloading tokenizer for {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            logger.info(f"Downloading model {model_name}")
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            logger.info(f"Successfully downloaded {model_name}")
            
        except Exception as e:
            logger.error(f"Error downloading {model_name}: {e}")
            continue
    
    logger.info("All models downloaded successfully!")

if __name__ == "__main__":
    main() 