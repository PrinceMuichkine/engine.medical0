import torch
import transformers
import tokenizers
import os, sys
from dataclasses import dataclass, field
import argparse
from PIL import Image

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def add_special_tokens_and_resize_model(tokenizer, model, vq_idx_nums=8192):
    if len(tokenizer.additional_special_tokens) != 0:
        return tokenizer.additional_special_tokens
    index_tokens = [f"<idx_{i}>" for i in range(vq_idx_nums)]
    special_tokens = {
        'additional_special_tokens': ['<start_index>'] + index_tokens + ['<end_index>'] + ['<pixel_newline>']
    }
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    
    return num_new_tokens

com_vision_args = argparse.Namespace(
    freeze_backbone=False,
    mm_patch_merge_type='flat',
    mm_projector_type='mlp2x_gelu',
    mm_use_im_patch_token=False,
    mm_use_im_start_end=False,
    mm_vision_select_feature='patch',
    mm_vision_select_layer=-2,
    model_name_or_path=None,
    pretrain_mm_mlp_adapter=None,
    tune_mm_mlp_adapter=False,
    version=None,
    vision_tower=None
)

gen_vision_args = argparse.Namespace(
    freeze_backbone=False,
    mm_patch_merge_type='flat',
    mm_projector_type='mlp2x_gelu',
    mm_use_im_patch_token=False,
    mm_use_im_start_end=False,
    mm_vision_select_feature='patch',
    mm_vision_select_layer=1,
    model_name_or_path=None,
    pretrain_mm_mlp_adapter=None,
    tune_mm_mlp_adapter=False,
    version=None,
    vision_tower=None
)

def resize_checkpoint_tensors(state_dict, model_state_dict):
    """
    Resize tensors in the checkpoint to match the model's shape
    This ensures that vocabulary size mismatches can be handled
    """
    modified = False
    
    # Check each key in the checkpoint
    for key in list(state_dict.keys()):
        # Skip if key not in model
        if key not in model_state_dict:
            continue
            
        # Check if shapes match
        if state_dict[key].shape != model_state_dict[key].shape:
            print(f"Resizing mismatch in key {key}: {state_dict[key].shape} vs {model_state_dict[key].shape}")
            
            # Handle embeddings and linear layers (2D tensors)
            if len(state_dict[key].shape) == 2 and len(model_state_dict[key].shape) == 2:
                old_rows, old_cols = state_dict[key].shape
                new_rows, new_cols = model_state_dict[key].shape
                
                # If rows differ (vocabulary size)
                if old_rows != new_rows:
                    print(f"Resizing rows: {old_rows} -> {new_rows}")
                    if new_rows > old_rows:
                        # Expand with zeros
                        new_tensor = torch.zeros_like(model_state_dict[key])
                        new_tensor[:old_rows, :] = state_dict[key]
                        state_dict[key] = new_tensor
                    else:
                        # Truncate
                        state_dict[key] = state_dict[key][:new_rows, :]
                        
                    modified = True
                
                # If columns differ (embedding dimension)
                if old_cols != new_cols:
                    print(f"Resizing columns: {old_cols} -> {new_cols}")
                    if new_cols > old_cols:
                        # Expand with zeros
                        new_tensor = torch.zeros_like(model_state_dict[key])
                        new_tensor[:, :old_cols] = state_dict[key]
                        state_dict[key] = new_tensor
                    else:
                        # Truncate
                        state_dict[key] = state_dict[key][:, :new_cols]
                        
                    modified = True
                    
    if modified:
        print("Some tensors were resized to match model dimensions")
    
    return state_dict

def load_weights(model, lora_path=None, fusion_layer_path=None):
    """
    Load checkpoint weights and adjust them if needed to fit model size
    """
    print(f"Loading weights from: {lora_path}")
    
    if lora_path is None or not os.path.exists(lora_path):
        print(f"H-LoRA checkpoint not found: {lora_path}")
        return model
    
    # Get model's state dict to compare shapes
    model_state_dict = model.state_dict()
    
    # Load lora weights
    lora_weights = torch.load(lora_path, map_location='cpu')

    # Resize weights if shapes don't match
    lora_weights = resize_checkpoint_tensors(lora_weights, model_state_dict)
    
    lora_unexpected_keys = model.load_state_dict(lora_weights, strict=False)[1]
    print("Unexpected keys in LoRA weights:", lora_unexpected_keys)
    
    # Check if we should load fusion layer weights
    if fusion_layer_path is not None and os.path.exists(fusion_layer_path):
        print(f"Loading fusion layer weights from: {fusion_layer_path}")
        fusion_layer_weights = torch.load(fusion_layer_path, map_location='cpu')
        
        # Resize fusion layer weights if needed
        fusion_layer_weights = resize_checkpoint_tensors(fusion_layer_weights, model_state_dict)
        
        fusion_layer_unexpected_keys = model.load_state_dict(fusion_layer_weights, strict=False)[1]
        print("Unexpected keys in fusion layer weights:", fusion_layer_unexpected_keys)
    
    return model

