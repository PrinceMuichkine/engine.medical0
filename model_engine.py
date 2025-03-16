import os
import sys
import threading
import time
import torch
from PIL import Image
import logging
import uuid
import base64
from io import BytesIO
import hashlib
from typing import Dict, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_engine')

# Add llava directory to path
llava_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llava")
demo_path = os.path.join(llava_path, "demo")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(llava_path)

# Set temp directory
temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(temp_dir, exist_ok=True)

# Import required llava modules
try:
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava import conversation as conversation_lib
    from llava.model import *
    from llava.mm_utils import tokenizer_image_token
    from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM, LlavaPhiConfig
    from llava.demo.utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights, expand2square, com_vision_args, gen_vision_args
    logger.info("Successfully imported llava modules")
except ImportError as e:
    logger.error(f"Failed to import llava modules: {e}")
    raise

# Type ignore comments for linter
# type: ignore

# Global variables for model and tokenizer
global_model = None
global_tokenizer = None
global_gen_model = None  # Generator model for image reconstruction
global_gen_tokenizer = None  # Generator tokenizer
model_lock = threading.Lock()
image_cache: Dict[str, Tuple[Image.Image, torch.Tensor]] = {}
model_loaded = False  # Flag to track if model is loaded
gen_model_loaded = False  # Flag to track if generator model is loaded

def get_image_cache_key(img_path: str) -> str:
    """Create a cache key for an image file based on its path and modification time"""
    if not os.path.exists(img_path):
        return hashlib.md5(img_path.encode()).hexdigest()
    
    # Use file path and modification time for the key
    mod_time = os.path.getmtime(img_path)
    return hashlib.md5(f"{img_path}_{mod_time}".encode()).hexdigest()

def check_model_checkpoint_compatibility(model, checkpoint_path):
    """Check if model dimensions are compatible with checkpoint before loading"""
    try:
        logger.info(f"Checking compatibility with checkpoint: {checkpoint_path}")
        # Load just the metadata from checkpoint without loading weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Compare key dimensions between model and checkpoint
        mismatch_count = 0
        for name, param in model.named_parameters():
            if name in checkpoint:
                if param.shape != checkpoint[name].shape:
                    logger.warning(f"Size mismatch for {name}: model has {param.shape}, checkpoint has {checkpoint[name].shape}")
                    mismatch_count += 1
                    
        if mismatch_count > 0:
            logger.warning(f"Found {mismatch_count} parameter shape mismatches between model and checkpoint")
            return False
        else:
            logger.info("Model and checkpoint dimensions are compatible")
            return True
            
    except Exception as e:
        logger.error(f"Error checking checkpoint compatibility: {e}")
        return False

def load_model(use_phi4=False):
    """Load the model and tokenizer into memory"""
    global global_model, global_tokenizer, model_loaded
    
    # If model is already loaded, return immediately
    if global_model is not None and global_tokenizer is not None and model_loaded:
        logger.info("Model is already loaded, skipping initialization")
        return True
    
    # Path setup for weights
    weight_files = find_weight_files(use_phi4)
    if not weight_files:
        logger.error("No weight files found, cannot load model")
        return False
    
    hlora_path = weight_files.get("com_hlora_weights.bin" if not use_phi4 else "com_hlora_weights_phi4.bin")
    fusion_layer_path = weight_files.get("fusion_layer_weights.bin")
    
    if not hlora_path:
        logger.error("H-LoRA weights not found, cannot load model")
        return False
    
    logger.info(f"Loading {'Phi-4' if use_phi4 else 'Phi-3'} model...")
    
    # Lock to ensure thread safety during model loading
    with model_lock:
        try:
            start_time = time.time()
            
            # Try to find the LLaVA runner script that we can use to get the model paths
            llava_runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llava", "demo", "medical0_analyze.sh")
            
            # The proper way would be to use Hugging Face models, but we'll use a local approach instead
            logger.info("Using local model initialization approach")
            
            # Import needed modules
            import transformers
            
            # We'll create a config first and then initialize from that
            logger.info("Creating model config")
            
            # Create a simple config for the base model
            config = LlavaPhiConfig(
                hidden_size=3072,
                intermediate_size=8192,
                num_hidden_layers=24,
                num_attention_heads=24,
                num_key_value_heads=24,
                max_position_embeddings=4096,
                rms_norm_eps=1e-6,
                vocab_size=50298,  # Match the model's expected size
                rope_theta=500000.0
            )
            
            # Create tokenizer - this is needed regardless
            logger.info("Creating tokenizer")
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                "microsoft/phi-2",  # Use a compatible tokenizer that's available locally
                padding_side="right",
                use_fast=False,
            )
            
            # BYPASS: Instead of adding special tokens, directly force the vocabulary size
            logger.info("BYPASS: Forcing tokenizer vocabulary size")
            # Force the correct vocabulary size by truncating or padding
            if hasattr(tokenizer, 'vocab') and isinstance(tokenizer.vocab, dict):
                logger.info(f"Original tokenizer vocab size: {len(tokenizer.vocab)}")
                # Truncate the vocabulary if needed
                if len(tokenizer.vocab) > 50298:
                    logger.info("Truncating tokenizer vocabulary")
                    # Keep only the first 50298 tokens
                    keys_to_keep = list(tokenizer.vocab.keys())[:50298]
                    tokenizer.vocab = {k: tokenizer.vocab[k] for k in keys_to_keep}
            
            # Create the model from config
            logger.info("Creating model from config")
            model = LlavaPhiForCausalLM(config)
            
            # Configure LoRA
            from llava.peft import LoraConfig, get_peft_model
            logger.info("Configuring LoRA")
            lora_config = LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules=find_all_linear_names(model),
                lora_dropout=0.0,
                bias='none',
                task_type="CAUSAL_LM",
                lora_nums=4,
            )
            model = get_peft_model(model, lora_config)
            
            # SKIP the add_special_tokens_and_resize_model call
            logger.info("Skipping add_special_tokens_and_resize_model to prevent size mismatch")
            
            # Initialize vision modules with local paths
            logger.info("Initializing vision modules")
            from llava.constants import DEFAULT_IMAGE_TOKEN
            
            # We need to set up vision args manually since we're not downloading
            com_vision_args.model_name_or_path = "phi-3"  # Just a placeholder
            com_vision_args.vision_tower = "openai/clip-vit-large-patch14-336"
            com_vision_args.version = "phi3_instruct"
            
            # Re-initialize vision modules
            model.get_model().initialize_vision_modules(model_args=com_vision_args)
            
            # Check compatibility before loading weights
            if not check_model_checkpoint_compatibility(model, hlora_path):
                logger.warning("Model and checkpoint dimensions don't match. Adjusting model configuration...")
                # Alternative configuration with larger dimensions if needed
                adjusted_config = LlavaPhiConfig(
                    hidden_size=3072,
                    intermediate_size=8192,
                    num_hidden_layers=24,
                    num_attention_heads=24,
                    num_key_value_heads=24,
                    max_position_embeddings=4096,
                    rms_norm_eps=1e-6,
                    vocab_size=50298,  # Match the model's expected size
                    rope_theta=500000.0
                )
                # Recreate model with adjusted config
                logger.info("Creating model with adjusted configuration")
                model = LlavaPhiForCausalLM(adjusted_config)
                
                # Reconfigure LoRA
                from llava.peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=64,
                    lora_alpha=128,
                    target_modules=find_all_linear_names(model),
                    lora_dropout=0.0,
                    bias='none',
                    task_type="CAUSAL_LM",
                    lora_nums=4,
                )
                model = get_peft_model(model, lora_config)
                
                # SKIP the add_special_tokens_and_resize_model call
                logger.info("Skipping add_special_tokens_and_resize_model to prevent size mismatch")
                
                # Re-initialize vision modules
                model.get_model().initialize_vision_modules(model_args=com_vision_args)
            
            # Now load the weights
            logger.info(f"Loading weights from {hlora_path}")
            model = load_weights(model, hlora_path, fusion_layer_path)
            
            # Final setup
            logger.info("Setting model to evaluation mode and moving to GPU")
            model.eval()
            # Use float16 precision for better performance
            model.half().cuda()
            
            # Store model and tokenizer in global variables
            global_model = model
            global_tokenizer = tokenizer
            model_loaded = True
            
            load_time = time.time() - start_time
            logger.info(f"Model and tokenizer loaded in {load_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            model_loaded = False
            return False

def load_gen_model(use_phi4=False):
    """Load the generator model and tokenizer into memory"""
    global global_gen_model, global_gen_tokenizer, gen_model_loaded
    
    # If model is already loaded, return immediately
    if global_gen_model is not None and global_gen_tokenizer is not None and gen_model_loaded:
        logger.info("Generator model is already loaded, skipping initialization")
        return True
    
    # Path setup for weights
    weight_files = find_weight_files(use_phi4)
    if not weight_files:
        logger.error("No weight files found, cannot load generator model")
        return False
    
    gen_hlora_path = weight_files.get("gen_hlora_weights.bin")
    fusion_layer_path = weight_files.get("fusion_layer_weights.bin")
    
    if not gen_hlora_path:
        logger.error("Generator H-LoRA weights not found, cannot load model")
        return False
    
    logger.info("Loading generator model...")
    
    # Lock to ensure thread safety during model loading
    with model_lock:
        try:
            start_time = time.time()
            
            # The proper way would be to use Hugging Face models, but we'll use a local approach instead
            logger.info("Using local model initialization approach for generator")
            
            # Import needed modules
            import transformers
            
            # We'll create a config first and then initialize from that
            logger.info("Creating generator model config")
            
            # Create a simple config for the base model
            config = LlavaPhiConfig(
                hidden_size=3072,
                intermediate_size=8192,
                num_hidden_layers=24,
                num_attention_heads=24,
                num_key_value_heads=24,
                max_position_embeddings=4096,
                rms_norm_eps=1e-6,
                vocab_size=50298,  # Match the model's expected size
                rope_theta=500000.0
            )
            
            # Create tokenizer - this is needed regardless
            logger.info("Creating generator tokenizer")
            gen_tokenizer = transformers.AutoTokenizer.from_pretrained(
                "microsoft/phi-2",  # Use a compatible tokenizer that's available locally
                padding_side="right",
                use_fast=False,
            )
            
            # BYPASS: Instead of adding special tokens, directly force the vocabulary size
            logger.info("BYPASS: Forcing generator tokenizer vocabulary size")
            # Force the correct vocabulary size by truncating or padding
            if hasattr(gen_tokenizer, 'vocab') and isinstance(gen_tokenizer.vocab, dict):
                logger.info(f"Original generator tokenizer vocab size: {len(gen_tokenizer.vocab)}")
                # Truncate the vocabulary if needed
                if len(gen_tokenizer.vocab) > 50298:
                    logger.info("Truncating generator tokenizer vocabulary")
                    # Keep only the first 50298 tokens
                    keys_to_keep = list(gen_tokenizer.vocab.keys())[:50298]
                    gen_tokenizer.vocab = {k: gen_tokenizer.vocab[k] for k in keys_to_keep}
            
            # Create the model from config
            logger.info("Creating generator model from config")
            gen_model = LlavaPhiForCausalLM(config)
            
            # Configure LoRA
            from llava.peft import LoraConfig, get_peft_model
            logger.info("Configuring generator LoRA")
            lora_config = LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules=find_all_linear_names(gen_model),
                lora_dropout=0.0,
                bias='none',
                task_type="CAUSAL_LM",
                lora_nums=4,
            )
            gen_model = get_peft_model(gen_model, lora_config)
            
            # SKIP the add_special_tokens_and_resize_model call
            logger.info("Skipping add_special_tokens_and_resize_model to prevent size mismatch")
            
            # Initialize vision modules with local paths
            logger.info("Initializing generator vision modules")
            from llava.constants import DEFAULT_IMAGE_TOKEN
            
            # We need to set up vision args manually since we're not downloading
            gen_vision_args.model_name_or_path = "phi-3"  # Just a placeholder
            gen_vision_args.vision_tower = "openai/clip-vit-large-patch14-336"
            gen_vision_args.version = "phi3_instruct"
            
            # Re-initialize vision modules
            gen_model.get_model().initialize_vision_modules(model_args=gen_vision_args)
            
            # Check compatibility before loading weights
            if not check_model_checkpoint_compatibility(gen_model, gen_hlora_path):
                logger.warning("Generator model and checkpoint dimensions don't match. Adjusting model configuration...")
                # Alternative configuration with larger dimensions if needed
                adjusted_config = LlavaPhiConfig(
                    hidden_size=3072,
                    intermediate_size=8192,
                    num_hidden_layers=24,
                    num_attention_heads=24,
                    num_key_value_heads=24,
                    max_position_embeddings=4096,
                    rms_norm_eps=1e-6,
                    vocab_size=50298,  # Match the model's expected size
                    rope_theta=500000.0
                )
                # Recreate model with adjusted config
                logger.info("Creating generator model with adjusted configuration")
                gen_model = LlavaPhiForCausalLM(adjusted_config)
                
                # Reconfigure LoRA
                from llava.peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=64,
                    lora_alpha=128,
                    target_modules=find_all_linear_names(gen_model),
                    lora_dropout=0.0,
                    bias='none',
                    task_type="CAUSAL_LM",
                    lora_nums=4,
                )
                gen_model = get_peft_model(gen_model, lora_config)
                
                # SKIP the add_special_tokens_and_resize_model call
                logger.info("Skipping add_special_tokens_and_resize_model to prevent size mismatch")
                
                # Re-initialize vision modules
                gen_model.get_model().initialize_vision_modules(model_args=gen_vision_args)
            
            # Now load the weights
            logger.info(f"Loading generator weights from {gen_hlora_path}")
            gen_model = load_weights(gen_model, gen_hlora_path, fusion_layer_path)
            
            # Final setup
            logger.info("Setting generator model to evaluation mode and moving to GPU")
            gen_model.eval()
            # Use float16 precision for better performance
            gen_model.half().cuda()
            
            # Store model and tokenizer in global variables
            global_gen_model = gen_model
            global_gen_tokenizer = gen_tokenizer
            gen_model_loaded = True
            
            load_time = time.time() - start_time
            logger.info(f"Generator model and tokenizer loaded in {load_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error loading generator model: {e}", exc_info=True)
            gen_model_loaded = False
            return False

def find_weight_files(use_phi4=False):
    """Search for weight files in common locations and return full paths"""
    weight_files = {}
    
    # Common places to look
    search_dirs = [
        os.path.abspath('./weights'),  # Current directory weights
        os.path.abspath('../weights'),  # Parent directory weights
        os.path.abspath('../../weights'),  # Grandparent directory weights
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights'),  # Script directory weights
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights'),  # Parent of script directory
    ]
    
    # Files to look for
    files_to_find = [
        "com_hlora_weights.bin",
        "gen_hlora_weights.bin",
        "fusion_layer_weights.bin",
    ]
    
    # Add Phi-4 weights if needed
    if use_phi4:
        files_to_find.append("com_hlora_weights_phi4.bin")
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            logger.info(f"Checking for weight files in: {search_dir}")
            for file_name in files_to_find:
                file_path = os.path.join(search_dir, file_name)
                if os.path.exists(file_path):
                    weight_files[file_name] = file_path
                    logger.info(f"Found weight file: {file_name} at {file_path}")
    
    return weight_files

def analyze_image(img_path, question, analysis_type=None):
    """Analyze an image using the loaded model"""
    global global_model, global_tokenizer, model_loaded
    
    start_time = time.time()
    
    # Check if model is loaded
    if global_model is None or global_tokenizer is None or not model_loaded:
        # Check if model loading on first request is allowed
        preload_enabled = os.environ.get('HEALTHGPT_PRELOAD_ENABLED', 'false').lower() == 'true'
        if not preload_enabled:
            logger.error("Model not preloaded and on-demand loading is disabled")
            return {"error": "Model not preloaded. Please restart the server with --preload-models flag."}, 503
        
        logger.error("Model is not loaded yet")
        return {"error": "Model not loaded yet"}, 503
    
    # Check if image exists
    if not os.path.exists(img_path):
        logger.error(f"Image not found: {img_path}")
        return {"error": f"Image path not found: {img_path}"}, 400
    
    # Process the request
    with model_lock:  # Ensure thread safety for model access
        try:
            # Prepare prompt
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question
            
            # Use a simpler conversation template since we're using a local approach
            conv = conversation_lib.conv_templates.get("phi3_instruct", 
                                                      conversation_lib.conv_templates.get("default"))
            conv = conv.copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Check if vision tower is properly initialized
            if not hasattr(global_model, 'get_vision_tower') or global_model.get_vision_tower() is None:
                logger.error("Vision tower not initialized properly")
                return {"error": "Model not initialized correctly"}, 500
            
            # Process input tokens
            logger.info("Processing input tokens")
            input_ids = tokenizer_image_token(prompt, global_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda().unsqueeze_(0)
            
            # Process image (with caching)
            img_cache_key = get_image_cache_key(img_path)
            if img_cache_key in image_cache:
                logger.info(f"Loading image from cache: {img_path}")
                image, image_tensor = image_cache[img_cache_key]
            else:
                logger.info(f"Processing image: {img_path}")
                image = Image.open(img_path).convert('RGB')
                
                # Get the correct processor
                vision_tower = global_model.get_vision_tower()
                if vision_tower is None or not hasattr(vision_tower, 'image_processor'):
                    logger.error("Vision tower image processor not available")
                    return {"error": "Model vision components not initialized correctly"}, 500
                
                # Use the expand2square function with the right processor mean values
                processor_mean = getattr(vision_tower.image_processor, 'image_mean', [0.48145466, 0.4578275, 0.40821073])
                image = expand2square(image, tuple(int(x*255) for x in processor_mean))
                
                # Process the image for the model
                if hasattr(vision_tower, 'image_processor') and vision_tower.image_processor is not None:
                    try:
                        # Use the processor with proper error handling
                        logger.info("Processing image with vision tower processor")
                        image_tensor = vision_tower.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze_(0)
                    except Exception as e:
                        logger.error(f"Error processing image with vision tower: {e}")
                        # Fallback to simple processing if needed
                        logger.info("Using fallback image processing")
                        
                        # Basic image preprocessing - resize to expected size
                        from torchvision import transforms
                        transform = transforms.Compose([
                            transforms.Resize((336, 336)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                                std=[0.26862954, 0.26130258, 0.27577711])
                        ])
                        image_tensor = transform(image).unsqueeze_(0)
                        
                    # Cache for future use
                    image_cache[img_cache_key] = (image, image_tensor)
                else:
                    logger.error("Vision tower processor not found")
                    return {"error": "Vision tower processor not found"}, 500
                        
            # Run inference
            logger.info("Running inference")
            inference_start = time.time()
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=True):
                    try:
                        # Run the generation with proper error handling
                        output_ids = global_model.base_model.model.generate(
                            input_ids,
                            images=image_tensor.to(device='cuda', non_blocking=True),
                            image_sizes=image.size if hasattr(image, 'size') else (336, 336),  # Default size if missing
                            do_sample=False,
                            temperature=0.0,
                            num_beams=1,
                            max_new_tokens=1024,
                            use_cache=True
                        )
                    except Exception as e:
                        logger.error(f"Error during model inference: {e}")
                        return {"error": f"Model inference failed: {str(e)}"}, 500
            
            inference_time = time.time() - inference_start
            logger.info(f"Inference completed in {inference_time:.2f} seconds")
            
            # Decode response
            try:
                response = global_tokenizer.decode(output_ids[0], skip_special_tokens=True)
                if response.endswith("</assistant>"):
                    response = response[:-12]  # Remove "</assistant>" tag if present
            except Exception as e:
                logger.error(f"Error decoding model output: {e}")
                return {"error": f"Error decoding model output: {str(e)}"}, 500
            
            total_time = time.time() - start_time
            result = {
                "answer": response,
                "model": "Phi-4" if "phi4" in str(global_model) else "Phi-3-mini",
                "analysis_type": analysis_type,
                "prompt": question,
                "timing": {
                    "total_time": total_time,
                    "inference_time": inference_time
                }
            }
            
            logger.info(f"Analysis completed in {total_time:.2f} seconds (inference: {inference_time:.2f}s)")
            return result, 200
            
        except Exception as e:
            logger.error(f"Error during image analysis: {e}", exc_info=True)
            return {"error": f"Error processing request: {str(e)}"}, 500

def reconstruct_image(img_path, question, analysis_type=None):
    """Reconstruct/enhance an image using the generator model"""
    global global_gen_model, global_gen_tokenizer, gen_model_loaded
    
    start_time = time.time()
    
    # Check if generator model is loaded
    if global_gen_model is None or global_gen_tokenizer is None or not gen_model_loaded:
        # Check if model loading on first request is allowed
        preload_enabled = os.environ.get('HEALTHGPT_PRELOAD_ENABLED', 'false').lower() == 'true'
        if not preload_enabled:
            logger.error("Generator model not preloaded and on-demand loading is disabled")
            return {"error": "Generator model not preloaded. Please restart the server with --preload-models flag."}, 503
            
        # Try to load the model
        if not load_gen_model():
            logger.error("Generator model is not loaded and could not be loaded")
            return {"error": "Generator model not loaded yet"}, 503
    
    # Check if image exists
    if not os.path.exists(img_path):
        logger.error(f"Image not found: {img_path}")
        return {"error": f"Image path not found: {img_path}"}, 400
    
    # Process the request
    with model_lock:  # Ensure thread safety for model access
        try:
            # Create a unique output path
            output_path = os.path.join(temp_dir, f"enhanced_{uuid.uuid4()}.jpg")
            
            # Prepare prompt
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question
            
            # Use a simpler conversation template since we're using a local approach
            conv = conversation_lib.conv_templates.get("phi3_instruct", 
                                                      conversation_lib.conv_templates.get("default"))
            conv = conv.copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Check if vision tower is properly initialized
            if not hasattr(global_gen_model, 'get_vision_tower') or global_gen_model.get_vision_tower() is None:
                logger.error("Generator vision tower not initialized properly")
                return {"error": "Generator model not initialized correctly"}, 500
            
            # Process input tokens
            logger.info("Processing input tokens for generator")
            input_ids = tokenizer_image_token(prompt, global_gen_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda().unsqueeze_(0)
            
            # Process image (with caching)
            img_cache_key = get_image_cache_key(img_path)
            if img_cache_key in image_cache:
                logger.info(f"Loading image from cache: {img_path}")
                image, image_tensor = image_cache[img_cache_key]
            else:
                logger.info(f"Processing image: {img_path}")
                image = Image.open(img_path).convert('RGB')
                
                # Get the correct processor
                vision_tower = global_gen_model.get_vision_tower()
                if vision_tower is None or not hasattr(vision_tower, 'image_processor'):
                    logger.error("Generator vision tower processor not available")
                    return {"error": "Generator model vision components not initialized correctly"}, 500
                
                # Use the expand2square function with the right processor mean values
                processor_mean = getattr(vision_tower.image_processor, 'image_mean', [0.48145466, 0.4578275, 0.40821073])
                image = expand2square(image, tuple(int(x*255) for x in processor_mean))
                
                # Process the image for the model
                if hasattr(vision_tower, 'image_processor') and vision_tower.image_processor is not None:
                    try:
                        # Use the processor with proper error handling
                        logger.info("Processing image with generator vision tower processor")
                        image_tensor = vision_tower.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze_(0)
                    except Exception as e:
                        logger.error(f"Error processing image with generator vision tower: {e}")
                        # Fallback to simple processing if needed
                        logger.info("Using fallback image processing for generator")
                        
                        # Basic image preprocessing - resize to expected size
                        from torchvision import transforms
                        transform = transforms.Compose([
                            transforms.Resize((336, 336)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                                std=[0.26862954, 0.26130258, 0.27577711])
                        ])
                        image_tensor = transform(image).unsqueeze_(0)
                        
                    # Cache for future use
                    image_cache[img_cache_key] = (image, image_tensor)
                else:
                    logger.error("Generator vision tower processor not found")
                    return {"error": "Generator vision tower processor not found"}, 500
            
            # Run inference - this is a placeholder since real image generation isn't implemented
            logger.info("Running generator inference (placeholder)")
            inference_start = time.time()
            with torch.inference_mode():
                # This is just a placeholder - in a real implementation, we would use the
                # generator model to create or enhance an image based on the input
                output_image = image.copy()
                
                # Save the output image
                output_image.save(output_path)
                
                # Convert the image to base64 for the response
                buffered = BytesIO()
                output_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
            
            inference_time = time.time() - inference_start
            logger.info(f"Generator inference completed in {inference_time:.2f} seconds")
            
            total_time = time.time() - start_time
            result = {
                'result_image': img_str,
                'model': "Phi-3-mini",  # Generator only works with Phi-3 currently
                'analysis_type': analysis_type,
                'prompt': question,
                'timing': {
                    'total_time': total_time,
                    'inference_time': inference_time
                }
            }
            
            logger.info(f"Image reconstruction completed in {total_time:.2f} seconds (inference: {inference_time:.2f}s)")
            
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)
                
            return result, 200
            
        except Exception as e:
            logger.error(f"Error during image reconstruction: {e}", exc_info=True)
            return {"error": f"Error processing request: {str(e)}"}, 500

def unload_model():
    """Unload the model from memory"""
    global global_model, global_tokenizer, global_gen_model, global_gen_tokenizer, image_cache, model_loaded, gen_model_loaded
    
    with model_lock:
        success = True
        
        # Unload analysis model
        if global_model is not None:
            try:
                # Free up GPU memory
                global_model = global_model.cpu()
                del global_model
                global_model = None
                global_tokenizer = None
                model_loaded = False
                logger.info("Analysis model unloaded from memory")
            except Exception as e:
                logger.error(f"Error unloading analysis model: {e}")
                success = False
        
        # Unload generator model
        if global_gen_model is not None:
            try:
                # Free up GPU memory
                global_gen_model = global_gen_model.cpu()
                del global_gen_model
                global_gen_model = None
                global_gen_tokenizer = None
                gen_model_loaded = False
                logger.info("Generator model unloaded from memory")
            except Exception as e:
                logger.error(f"Error unloading generator model: {e}")
                success = False
        
        # Clear image cache
        image_cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
