import os
import torch
import time
import shutil
from diffusers import FluxPipeline
from optimum.quanto import freeze, qint8, quantize

# Constants
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/files.tar"  # Updated URL
MODEL_CACHE = "FLUX.1-schnell"
HALF_PRECISION_CACHE = "FLUX.1-schnell-fp16"  # Use existing FP16 model if available
INT8_CACHE = "FLUX.1-schnell-int8"

def download_weights(url, dest):
    """Download weights if they don't exist"""
    if not os.path.exists(dest):
        print(f"Downloading weights from {url} to {dest}...")
        os.system(f"pget {url} {dest}")
    else:
        print(f"Weights already exist at {dest}")

def get_dir_size(path):
    """Get directory size in GB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024 * 1024)  # Convert to GB

def main():
    # Check if int8 model already exists
    if os.path.exists(INT8_CACHE):
        print(f"INT8 model already exists at {INT8_CACHE}")
        return
    
    # Determine source model - prefer using FP16 if available
    if os.path.exists(HALF_PRECISION_CACHE):
        source_model = HALF_PRECISION_CACHE
        print(f"Using existing half-precision model from {HALF_PRECISION_CACHE}")
    else:
        # Download original model if needed
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        source_model = MODEL_CACHE
    
    # Measure original model size
    original_size = get_dir_size(source_model)
    print(f"Source model size: {original_size:.2f} GB")
    
    # Load the model in half precision
    print(f"Loading model from {source_model}...")
    start_time = time.time()
    
    pipe = FluxPipeline.from_pretrained(
        source_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True  # Force using local files only
    )
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Quantize the model components to INT8
    print("Quantizing model to INT8...")
    start_time = time.time()
    
    # Quantize UNet (the main diffusion model)
    print("Quantizing UNet...")
    quantize(pipe.unet, weights=qint8)
    freeze(pipe.unet)
    
    # Quantize text encoder
    print("Quantizing text encoder...")
    quantize(pipe.text_encoder, weights=qint8)
    freeze(pipe.text_encoder)
    
    # Quantize VAE decoder (for image generation)
    print("Quantizing VAE decoder...")
    quantize(pipe.vae.decoder, weights=qint8)
    freeze(pipe.vae.decoder)
    
    quant_time = time.time() - start_time
    print(f"Model quantized in {quant_time:.2f} seconds")
    
    # Save the quantized model
    print(f"Saving quantized model to {INT8_CACHE}...")
    start_time = time.time()
    pipe.save_pretrained(INT8_CACHE)
    save_time = time.time() - start_time
    print(f"Model saved in {save_time:.2f} seconds")
    
    # Measure quantized model size
    quantized_size = get_dir_size(INT8_CACHE)
    print(f"Quantized model size: {quantized_size:.2f} GB")
    print(f"Size reduction: {(1 - quantized_size / original_size) * 100:.2f}%")

if __name__ == "__main__":
    main()
