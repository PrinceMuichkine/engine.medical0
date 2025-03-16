#!/usr/bin/env python3
"""
Script to modify checkpoints to match model vocabulary sizes.
This allows us to work around size mismatch issues.
"""

import os
import sys
import torch
import argparse
import logging
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('checkpoint_modifier')

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
    
    # Check if this is a raw state dict or a more complex nested structure
    modified = False
    
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
    
    # Find the actual keys in the checkpoint
    embed_keys = [k for k in checkpoint.keys() if any(candidate in k for candidate in embed_key_candidates)]
    lm_head_keys = [k for k in checkpoint.keys() if any(candidate in k for candidate in lm_head_key_candidates)]
    
    if not embed_keys and not lm_head_keys:
        logger.warning("Could not find embedding or LM head keys in checkpoint")
        logger.info("Available keys: " + ", ".join(list(checkpoint.keys())[:10]) + "...")
        return False
    
    # Modify each found key
    for key in embed_keys:
        logger.info(f"Modifying embedding key: {key}")
        if key in checkpoint:
            checkpoint[key] = resize_embedding_weight(
                checkpoint[key], target_vocab_size, random_init=random_init
            )
            modified = True
        else:
            logger.warning(f"Key {key} not found in checkpoint")
    
    for key in lm_head_keys:
        logger.info(f"Modifying LM head key: {key}")
        if key in checkpoint:
            checkpoint[key] = resize_embedding_weight(
                checkpoint[key], target_vocab_size, random_init=random_init
            )
            modified = True
        else:
            logger.warning(f"Key {key} not found in checkpoint")
    
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

def main():
    parser = argparse.ArgumentParser(description="Modify model checkpoint vocabulary size")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint file")
    parser.add_argument("--output", help="Path to save the modified checkpoint (default: adds '_modified' suffix)")
    parser.add_argument("--target-vocab-size", type=int, required=True, help="Target vocabulary size")
    parser.add_argument("--random-init", action="store_true", help="Initialize new tokens with random values")
    parser.add_argument("--backup", action="store_true", help="Create a backup of the original checkpoint")
    
    args = parser.parse_args()
    
    # Default output path if not provided
    if not args.output:
        filename, ext = os.path.splitext(args.checkpoint)
        args.output = f"{filename}_modified{ext}"
    
    # Create backup if requested
    if args.backup:
        backup_path = f"{args.checkpoint}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Creating backup at {backup_path}")
        try:
            shutil.copy2(args.checkpoint, backup_path)
            logger.info("Backup created successfully")
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return 1
    
    # Modify checkpoint
    success = modify_checkpoint(
        args.checkpoint, 
        args.output, 
        args.target_vocab_size,
        args.random_init
    )
    
    if success:
        logger.info("Checkpoint modification complete")
        return 0
    else:
        logger.error("Checkpoint modification failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 