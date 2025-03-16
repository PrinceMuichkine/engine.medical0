import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, Any, Tuple
import torch
import transformers
import tokenizers
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM, LlavaPhiConfig
from PIL import Image
import pickle
import argparse
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
from utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights, expand2square
import time
import hashlib

# Cache directory for model artifacts
os.makedirs(os.path.expanduser("~/.cache/healthgpt"), exist_ok=True)
CACHE_DIR = os.path.expanduser("~/.cache/healthgpt")

# Global cache for model components
_MODEL_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}
_IMAGE_CACHE: Dict[str, Tuple[Image.Image, torch.Tensor]] = {}

def get_cache_key(args):
    """Generate a unique cache key based on model arguments"""
    key_components = [
        args.model_name_or_path,
        args.dtype,
        args.hlora_r,
        args.hlora_alpha,
        args.hlora_nums,
        args.vq_idx_nums,
        args.vit_path,
        args.hlora_path,
        args.fusion_layer_path
    ]
    return hashlib.md5(str(key_components).encode()).hexdigest()

def get_image_cache_key(img_path):
    """Generate a cache key for images based on path and modification time"""
    if not os.path.exists(img_path):
        return None
    mtime = os.path.getmtime(img_path)
    return f"{img_path}_{mtime}"

def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/Phi-3-mini-4k-instruct')
    parser.add_argument('--dtype', type=str, default='FP32')
    parser.add_argument('--attn_implementation', type=str, default=None)
    parser.add_argument('--hlora_r', type=int, default=16)
    parser.add_argument('--hlora_alpha', type=int, default=32)
    parser.add_argument('--hlora_dropout', type=float, default=0.0)
    parser.add_argument('--hlora_nums', type=int, default=4)
    parser.add_argument('--vq_idx_nums', type=int, default=1024)
    parser.add_argument('--instruct_template', type=str, default='phi3_instruct')
    parser.add_argument('--vit_path', type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument('--hlora_path', type=str, default=None)
    parser.add_argument('--fusion_layer_path', type=str, default=None)
    parser.add_argument('--question', type=str, default=None)
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--use_cache', type=bool, default=True)
    
    args = parser.parse_args()
    
    start_time = time.time()
    model_dtype = torch.float32 if args.dtype == 'FP32' else (torch.float16 if args.dtype == 'FP16' else torch.bfloat16)
    
    # Generate cache keys for model components
    cache_key = get_cache_key(args)
    
    # Try to load model and tokenizer from cache
    model = None
    tokenizer = None
    
    if args.use_cache and cache_key in _MODEL_CACHE:
        print(f"Loading model from cache (key: {cache_key[:8]}...)")
        model, tokenizer = _MODEL_CACHE[cache_key], _TOKENIZER_CACHE[cache_key]
    else:
        print("Initializing model from scratch...")
        model = LlavaPhiForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            attn_implementation=args.attn_implementation,
            torch_dtype=model_dtype
        )

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

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            padding_side="right",
            use_fast=False,
        )
        num_new_tokens = add_special_tokens_and_resize_model(tokenizer, model, args.vq_idx_nums)
        print(f"Number of new tokens added for unified task: {num_new_tokens}")

        from utils import com_vision_args
        com_vision_args.model_name_or_path = args.model_name_or_path
        com_vision_args.vision_tower = args.vit_path
        com_vision_args.version = args.instruct_template

        model.get_model().initialize_vision_modules(model_args=com_vision_args)
        model.get_vision_tower().to(dtype=model_dtype)

        model = load_weights(model, args.hlora_path)
        model.eval()
        model.to(model_dtype).cuda()
        
        # Cache the model and tokenizer if caching is enabled
        if args.use_cache:
            _MODEL_CACHE[cache_key] = model
            _TOKENIZER_CACHE[cache_key] = tokenizer
            print(f"Model cached with key: {cache_key[:8]}...")

    model_load_time = time.time()
    print(f"Model preparation time: {model_load_time - start_time:.2f} seconds")

    question = args.question
    img_path = args.img_path

    if img_path:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question
    else:
        qs = question
    conv = conversation_lib.conv_templates[args.instruct_template].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda().unsqueeze_(0)
    
    image_tensor = None
    image = None
    
    if img_path:
        # Try to load processed image from cache
        img_cache_key = get_image_cache_key(img_path)
        
        if args.use_cache and img_cache_key in _IMAGE_CACHE:
            print(f"Loading image from cache: {img_path}")
            image, image_tensor = _IMAGE_CACHE[img_cache_key]
        else:
            print(f"Processing image: {img_path}")
            image = Image.open(img_path).convert('RGB')
            image = expand2square(image, tuple(int(x*255) for x in model.get_vision_tower().image_processor.image_mean))
            image_tensor = model.get_vision_tower().image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze_(0)
            
            # Cache the processed image
            if args.use_cache:
                _IMAGE_CACHE[img_cache_key] = (image, image_tensor)
    
    img_process_time = time.time()
    print(f"Image processing time: {img_process_time - model_load_time:.2f} seconds")

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=True, dtype=model_dtype):
            output_ids = model.base_model.model.generate(
            input_ids,
            images=image_tensor.to(dtype=model_dtype, device='cuda', non_blocking=True) if img_path else None,
            image_sizes=image.size if img_path else None,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True)
    
    inference_time = time.time()
    print(f"Inference time: {inference_time - img_process_time:.2f} seconds")
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)[:-8]
    print(f'Q: {question}')
    print(f'HealthGPT: {response}')
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")


if __name__ == "__main__":
    infer()