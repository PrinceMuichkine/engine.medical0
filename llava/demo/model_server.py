import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import argparse
import time
import hashlib
import threading
import json
from flask import Flask, request, jsonify
import torch
import transformers
from PIL import Image

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM
from utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights, expand2square

# Global model variables
global_model = None
global_tokenizer = None
model_lock = threading.Lock()

app = Flask(__name__)

def load_model(args):
    """Load the model and tokenizer once at startup"""
    global global_model, global_tokenizer
    
    print("Loading model and tokenizer...")
    start_time = time.time()
    
    # Set model dtype
    model_dtype = torch.float32 if args.dtype == 'FP32' else (torch.float16 if args.dtype == 'FP16' else torch.bfloat16)
    
    # Load model
    model = LlavaPhiForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=model_dtype
    )
    
    # Configure LoRA
    from llava.peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=args.hlora_r,
        lora_alpha=args.hlora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=args.hlora_dropout,
        bias='none',
        task_type="CAUSAL_LM",
        lora_nums=args.hlora_nums,
    )
    model = get_peft_model(model, lora_config)
    
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
    
    # Add special tokens
    num_new_tokens = add_special_tokens_and_resize_model(tokenizer, model, args.vq_idx_nums)
    print(f"Number of new tokens added for unified task: {num_new_tokens}")
    
    # Initialize vision modules
    from utils import com_vision_args
    com_vision_args.model_name_or_path = args.model_name_or_path
    com_vision_args.vision_tower = args.vit_path
    com_vision_args.version = args.instruct_template
    
    model.get_model().initialize_vision_modules(model_args=com_vision_args)
    model.get_vision_tower().to(dtype=model_dtype)
    
    # Load weights
    model = load_weights(model, args.hlora_path)
    model.eval()
    model.to(model_dtype).cuda()
    
    # Store model and tokenizer in global variables
    global_model = model
    global_tokenizer = tokenizer
    
    load_time = time.time() - start_time
    print(f"Model and tokenizer loaded in {load_time:.2f} seconds")

# Cache for processed images
image_cache = {}

def get_image_cache_key(img_path):
    """Generate a cache key for images based on path and modification time"""
    if not os.path.exists(img_path):
        return None
    mtime = os.path.getmtime(img_path)
    return f"{img_path}_{mtime}"

@app.route('/api/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({"status": "ok", "model_loaded": global_model is not None})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    start_time = time.time()
    
    if global_model is None or global_tokenizer is None:
        return jsonify({"error": "Model not loaded yet"}), 503
    
    # Get request data
    data = request.json
    img_path = data.get('img_path')
    question = data.get('question', "Analyze this medical image.")
    
    if not img_path or not os.path.exists(img_path):
        return jsonify({"error": f"Image path not found: {img_path}"}), 400
    
    # Process the request
    with model_lock:  # Ensure thread safety for model access
        try:
            # Prepare prompt
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv = conversation_lib.conv_templates["phi3_instruct"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, global_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda().unsqueeze_(0)
            
            # Process image (with caching)
            img_cache_key = get_image_cache_key(img_path)
            if img_cache_key in image_cache:
                print(f"Loading image from cache: {img_path}")
                image, image_tensor = image_cache[img_cache_key]
            else:
                print(f"Processing image: {img_path}")
                image = Image.open(img_path).convert('RGB')
                image = expand2square(image, tuple(int(x*255) for x in global_model.get_vision_tower().image_processor.image_mean))
                image_tensor = global_model.get_vision_tower().image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze_(0)
                image_cache[img_cache_key] = (image, image_tensor)
            
            # Run inference
            inference_start = time.time()
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=True):
                    output_ids = global_model.base_model.model.generate(
                        input_ids,
                        images=image_tensor.to(device='cuda', non_blocking=True),
                        image_sizes=image.size,
                        do_sample=False,
                        temperature=0.0,
                        num_beams=1,
                        max_new_tokens=1024,
                        use_cache=True
                    )
            
            inference_time = time.time() - inference_start
            
            # Decode response
            response = global_tokenizer.decode(output_ids[0], skip_special_tokens=True)[:-8]
            
            total_time = time.time() - start_time
            result = {
                "response": response,
                "timing": {
                    "total_time": total_time,
                    "inference_time": inference_time
                }
            }
            return jsonify(result)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

def run_server(args):
    # Load model before starting server
    load_model(args)
    
    # Start Flask server
    port = args.port
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/Phi-3-mini-4k-instruct')
    parser.add_argument('--dtype', type=str, default='FP16')
    parser.add_argument('--attn_implementation', type=str, default=None)
    parser.add_argument('--hlora_r', type=int, default=64)
    parser.add_argument('--hlora_alpha', type=int, default=128)
    parser.add_argument('--hlora_dropout', type=float, default=0.0)
    parser.add_argument('--hlora_nums', type=int, default=4)
    parser.add_argument('--vq_idx_nums', type=int, default=8192)
    parser.add_argument('--instruct_template', type=str, default='phi3_instruct')
    parser.add_argument('--vit_path', type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument('--hlora_path', type=str, default='../../weights/com_hlora_weights.bin')
    parser.add_argument('--fusion_layer_path', type=str, default='../../weights/fusion_layer_weights.bin')
    parser.add_argument('--port', type=int, default=5000)
    
    args = parser.parse_args()
    run_server(args) 