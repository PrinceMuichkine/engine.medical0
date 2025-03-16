#!/usr/bin/env python3
"""
Script to automatically fix model size mismatches by either:
1. Modifying the checkpoint to match the model size
2. Modifying the model configuration to match the checkpoint size

This script will detect model checkpoints and fix the vocabulary size issues.
"""

import os
import sys
import torch
import logging
import argparse
import glob
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_size_fixer')

def find_checkpoints(weights_dir=None):
    """Find all potential checkpoint files in the weights directory"""
    if weights_dir is None:
        # Common places to look for weights
        search_dirs = [
            os.path.abspath('./weights'),  # Current directory weights
            os.path.abspath('../weights'),  # Parent directory weights
            os.path.abspath('../../weights'),  # Grandparent directory weights
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights'),  # Script directory weights
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights'),  # Parent of script
        ]
    else:
        search_dirs = [os.path.abspath(weights_dir)]
    
    checkpoint_files = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            logger.info(f"Searching for checkpoint files in: {search_dir}")
            # Look for .bin, .pt, .pth files
            for ext in ['*.bin', '*.pt', '*.pth']:
                files = glob.glob(os.path.join(search_dir, ext))
                for f in files:
                    checkpoint_files.append(f)
    
    logger.info(f"Found {len(checkpoint_files)} potential checkpoint files")
    return checkpoint_files

def get_checkpoint_vocab_size(checkpoint_path):
    """
    Get the vocabulary size from a checkpoint
    Returns (vocab_size, key_name) if found, else (None, None)
    """
    try:
        logger.info(f"Loading checkpoint to detect vocabulary size: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Track embed and lm_head keys to check
        embed_key_candidates = [
            "base_model.model.model.embed_tokens.weight", 
            "model.model.embed_tokens.weight",
            "transformer.wte.weight",
            "embed_tokens.weight",
            "model.embed_tokens.weight"
        ]
        
        lm_head_key_candidates = [
            "base_model.model.lm_head.weight", 
            "model.lm_head.weight",
            "lm_head.weight"
        ]
        
        # Find the first matching key
        for key in embed_key_candidates + lm_head_key_candidates:
            if key in checkpoint:
                vocab_size = checkpoint[key].shape[0]
                logger.info(f"Found vocabulary size {vocab_size} in key {key}")
                return vocab_size, key
        
        # If exact match not found, try partial match
        for key in checkpoint.keys():
            for candidate in embed_key_candidates + lm_head_key_candidates:
                if candidate in key:
                    vocab_size = checkpoint[key].shape[0]
                    logger.info(f"Found vocabulary size {vocab_size} in partially matching key {key}")
                    return vocab_size, key
        
        logger.warning(f"Could not find vocabulary size in checkpoint: {checkpoint_path}")
        return None, None
    
    except Exception as e:
        logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None, None

def resize_embedding_weight(weight, new_num_tokens, padding_idx=None, random_init=False):
    """
    Resize embedding layer to match new vocabulary size
    """
    logger.info(f"Resizing embedding from {weight.shape[0]} to {new_num_tokens}")
    
    old_num_tokens, embedding_dim = weight.shape
    
    if new_num_tokens == old_num_tokens:
        return weight
    
    # Create new embedding
    if new_num_tokens > old_num_tokens:
        # Initialize expanded part
        if random_init:
            # Initialize with random values, matching the distribution of the existing weights
            logger.info("Using random initialization for expanded vocabulary")
            mean = weight.mean().item()
            std = weight.std().item()
            extra_tokens = new_num_tokens - old_num_tokens
            new_part = torch.normal(mean=mean, std=std, size=(extra_tokens, embedding_dim))
            new_weight = torch.cat([weight, new_part], dim=0)
        else:
            # Initialize expanded part with zeros
            logger.info("Using zero initialization for expanded vocabulary")
            new_weight = torch.zeros(new_num_tokens, embedding_dim)
            new_weight[:old_num_tokens, :] = weight
            
            # Initialize padding idx embedding (if specified) to zeros
            if padding_idx is not None and padding_idx < new_num_tokens:
                with torch.no_grad():
                    new_weight[padding_idx].fill_(0)
                    
    else:
        # Truncate
        logger.info("Truncating embedding to smaller vocabulary")
        new_weight = weight[:new_num_tokens, :]
        
    return new_weight

def modify_checkpoint(checkpoint_path, output_path, target_vocab_size, random_init=False):
    """
    Modify checkpoint to match target vocabulary size
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return False
    
    logger.info("Checkpoint loaded successfully")
    
    # Track embed and lm_head keys to modify
    embed_key_candidates = [
        "base_model.model.model.embed_tokens.weight", 
        "model.model.embed_tokens.weight",
        "transformer.wte.weight",
        "embed_tokens.weight",
        "model.embed_tokens.weight"
    ]
    
    lm_head_key_candidates = [
        "base_model.model.lm_head.weight", 
        "model.lm_head.weight",
        "lm_head.weight",
        "transformer.wte.weight"  # Some models tie weights
    ]
    
    # Find the actual keys in the checkpoint (including partial matches)
    all_keys = list(checkpoint.keys())
    embed_keys = []
    lm_head_keys = []
    
    for key in all_keys:
        if any(candidate in key for candidate in embed_key_candidates):
            embed_keys.append(key)
        if any(candidate in key for candidate in lm_head_key_candidates):
            lm_head_keys.append(key)
    
    if not embed_keys and not lm_head_keys:
        logger.warning("Could not find embedding or LM head keys in checkpoint")
        logger.info("Available keys: " + ", ".join(list(checkpoint.keys())[:10]) + "...")
        return False
    
    modified = False
    
    # Modify each found key
    for key in embed_keys:
        logger.info(f"Modifying embedding key: {key}")
        if key in checkpoint and isinstance(checkpoint[key], torch.Tensor) and len(checkpoint[key].shape) == 2:
            checkpoint[key] = resize_embedding_weight(
                checkpoint[key], target_vocab_size, random_init=random_init
            )
            modified = True
        else:
            logger.warning(f"Key {key} not found or not a 2D tensor")
    
    for key in lm_head_keys:
        logger.info(f"Modifying LM head key: {key}")
        if key in checkpoint and isinstance(checkpoint[key], torch.Tensor) and len(checkpoint[key].shape) == 2:
            checkpoint[key] = resize_embedding_weight(
                checkpoint[key], target_vocab_size, random_init=random_init
            )
            modified = True
        else:
            logger.warning(f"Key {key} not found or not a 2D tensor")
    
    if not modified:
        logger.error("No keys were modified. Checkpoint remains unchanged.")
        return False
    
    # Save the modified checkpoint
    logger.info(f"Saving modified checkpoint to {output_path}")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(checkpoint, output_path)
        logger.info("Checkpoint saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        return False

def fix_model_engine_config(target_vocab_size):
    """
    Fix the model_engine.py configuration to use the specified vocabulary size
    """
    model_engine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_engine.py")
    if not os.path.exists(model_engine_path):
        logger.error(f"Could not find model_engine.py at {model_engine_path}")
        return False
    
    # Create backup
    backup_path = f"{model_engine_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Creating backup of model_engine.py at {backup_path}")
    try:
        shutil.copy2(model_engine_path, backup_path)
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return False
    
    # Read the file
    with open(model_engine_path, 'r') as f:
        content = f.read()
    
    # Replace vocab_size in config
    import re
    pattern = r'vocab_size=\d+'
    replacement = f'vocab_size={target_vocab_size}'
    new_content = re.sub(pattern, replacement, content)
    
    # Update add_special_tokens_and_resize_model call
    pattern = r'add_special_tokens_and_resize_model\(.*?, \d+\)'
    replacement = f'add_special_tokens_and_resize_model(tokenizer, model, 0)'
    new_content = re.sub(pattern, replacement, new_content)
    
    # Write back to file
    with open(model_engine_path, 'w') as f:
        f.write(new_content)
    
    logger.info(f"Updated model_engine.py to use vocabulary size {target_vocab_size}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Fix model size mismatches")
    parser.add_argument("--weights-dir", help="Directory containing weight files")
    parser.add_argument("--target-vocab-size", type=int, help="Target vocabulary size (default: auto-detect)")
    parser.add_argument("--fix-method", choices=['checkpoint', 'config', 'both'], default='both',
                        help="What to fix: 'checkpoint' modifies checkpoints, 'config' modifies model_engine.py, 'both' does both")
    parser.add_argument("--random-init", action="store_true", help="Initialize new tokens with random values")
    parser.add_argument("--force", action="store_true", help="Force fix even if no mismatch detected")
    
    args = parser.parse_args()
    
    # Step 1: Find all checkpoint files
    checkpoint_files = find_checkpoints(args.weights_dir)
    if not checkpoint_files:
        logger.error("No checkpoint files found")
        return 1
    
    # Detect target vocabulary size from the first checkpoint if not specified
    target_vocab_size = args.target_vocab_size
    if target_vocab_size is None:
        # Try to detect from the first checkpoint that we can load
        for checkpoint_path in checkpoint_files:
            vocab_size, key = get_checkpoint_vocab_size(checkpoint_path)
            if vocab_size is not None:
                target_vocab_size = vocab_size
                logger.info(f"Auto-detected vocabulary size: {target_vocab_size}")
                break
        
        if target_vocab_size is None:
            logger.error("Could not auto-detect vocabulary size and none specified")
            return 1
    
    # Fix model_engine.py configuration if requested
    if args.fix_method in ['config', 'both']:
        logger.info(f"Fixing model_engine.py to use vocabulary size {target_vocab_size}")
        success = fix_model_engine_config(target_vocab_size)
        if not success:
            logger.error("Failed to fix model_engine.py")
            if args.fix_method == 'config':
                return 1
    
    # Fix checkpoints if requested
    if args.fix_method in ['checkpoint', 'both']:
        logger.info(f"Fixing checkpoints to use vocabulary size {target_vocab_size}")
        all_success = True
        
        for checkpoint_path in checkpoint_files:
            # Skip if file doesn't exist or is too small to be a real checkpoint
            if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) < 1000:
                continue
            
            try:
                # Check if this checkpoint needs fixing
                current_vocab_size, _ = get_checkpoint_vocab_size(checkpoint_path)
                if current_vocab_size is None:
                    logger.warning(f"Could not detect vocabulary size for {checkpoint_path}, skipping")
                    continue
                
                if current_vocab_size == target_vocab_size and not args.force:
                    logger.info(f"Checkpoint {checkpoint_path} already has the target vocabulary size, skipping")
                    continue
                
                # Create output path
                output_path = checkpoint_path.replace('.bin', f'_vocab{target_vocab_size}.bin')
                if output_path == checkpoint_path:
                    # If extension wasn't .bin, append suffix
                    filename, ext = os.path.splitext(checkpoint_path)
                    output_path = f"{filename}_vocab{target_vocab_size}{ext}"
                
                # Fix checkpoint
                success = modify_checkpoint(
                    checkpoint_path, 
                    output_path, 
                    target_vocab_size,
                    args.random_init
                )
                
                if success:
                    logger.info(f"Successfully fixed checkpoint: {checkpoint_path} -> {output_path}")
                else:
                    logger.error(f"Failed to fix checkpoint: {checkpoint_path}")
                    all_success = False
            
            except Exception as e:
                logger.error(f"Error processing checkpoint {checkpoint_path}: {e}")
                all_success = False
        
        if not all_success:
            logger.warning("Some checkpoints could not be fixed")
    
    logger.info("Model size fixing complete")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 