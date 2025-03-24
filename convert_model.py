#!/usr/bin/env python3
# Script to convert the model to half precision

import os
import time
import torch
from diffusers import FluxPipeline

MODEL_CACHE = "FLUX.1-schnell"
HALF_PRECISION_CACHE = "FLUX.1-schnell-fp16"

def save_model_half_precision(model_path, output_path):
    """Save the model in half precision to save space"""
    start = time.time()
    print(f"Loading model from {model_path} to convert to half precision...")
    
    # Load the model in half precision
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use half precision during loading
        low_cpu_mem_usage=True,
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save the model in half precision
    print(f"Saving model to {output_path} in half precision...")
    pipe.save_pretrained(output_path, safe_serialization=True)
    
    # Calculate sizes
    original_size = sum(os.path.getsize(os.path.join(model_path, f)) 
                        for f in os.listdir(model_path) 
                        if os.path.isfile(os.path.join(model_path, f)))
    
    new_size = sum(os.path.getsize(os.path.join(output_path, f)) 
                  for f in os.listdir(output_path) 
                  if os.path.isfile(os.path.join(output_path, f)))
    
    print(f"Original model size: {original_size / (1024*1024):.2f} MB")
    print(f"Half precision model size: {new_size / (1024*1024):.2f} MB")
    print(f"Space saved: {(original_size - new_size) / (1024*1024):.2f} MB ({100 * (1 - new_size/original_size):.2f}%)")
    print(f"Model conversion and saving took: {time.time() - start:.2f} seconds")
    return True

if __name__ == "__main__":
    if not os.path.exists(MODEL_CACHE):
        print(f"Error: Original model path {MODEL_CACHE} does not exist!")
        exit(1)
        
    if os.path.exists(HALF_PRECISION_CACHE):
        print(f"Half precision model already exists at {HALF_PRECISION_CACHE}")
        print("Do you want to overwrite it? (y/n)")
        choice = input().lower()
        if choice != 'y':
            print("Aborting conversion.")
            exit(0)
    
    save_model_half_precision(MODEL_CACHE, HALF_PRECISION_CACHE)
    print("Conversion completed successfully!")
