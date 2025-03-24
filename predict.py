# Prediction interface for Cog 
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
import numpy as np
from typing import List, Union
from diffusers import FluxPipeline
from PIL import Image

MODEL_CACHE = "FLUX.1-schnell"
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/files.tar"
HALF_PRECISION_CACHE = "FLUX.1-schnell-fp16"  # New location for half precision model

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()

        # Check if half precision model exists
        if os.path.exists(HALF_PRECISION_CACHE):
            print(f"Loading half precision model from {HALF_PRECISION_CACHE}")
            model_path = HALF_PRECISION_CACHE
        else:
            # Download the model weights if they don't exist
            if not os.path.exists(MODEL_CACHE):
                download_weights(MODEL_URL, MODEL_CACHE)
            
            # Use original model path
            print(f"Using original model from {MODEL_CACHE}")
            model_path = MODEL_CACHE
        
        # Enable TensorFloat-32 for faster computation on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Set memory optimization settings
        torch.cuda.empty_cache()
        
        # Load the model with memory optimizations
        print(f"Loading Flux.schnell Pipeline from {model_path}")
        self.pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use half precision
            low_cpu_mem_usage=True,     # Optimize CPU memory usage during loading
        )
        
        # Move to CUDA in a memory-efficient way
        print("Moving model to CUDA...")
        self.pipe.to("cuda")
        
        # Add hooks to log tensor shapes during forward pass
        self.tensor_shapes = {}
        
        def log_shape_hook(name):
            def hook(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    if isinstance(input[0], torch.Tensor):
                        self.tensor_shapes[f"{name}_input"] = input[0].shape
                if isinstance(output, torch.Tensor):
                    self.tensor_shapes[f"{name}_output"] = output.shape
            return hook
        
        # Register hooks on key components
        if hasattr(self.pipe, "transformer"):
            self.pipe.transformer.register_forward_hook(log_shape_hook("transformer"))
            if hasattr(self.pipe.transformer, "x_embedder"):
                self.pipe.transformer.x_embedder.register_forward_hook(log_shape_hook("x_embedder"))
        
        print(f"setup took: ", time.time() - start)

    @staticmethod
    def make_multiple_of_16(n):
        """Make sure width and height are multiples of 16 for the model"""
        return ((n + 15) // 16) * 16

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for generated image"),
        width: int = Input(description="Width of the generated image", default=768),
        height: int = Input(description="Height of the generated image", default=768),
        seed1: int = Input(description="First random seed for interpolation", default=1234567890),
        seed2: int = Input(description="Second random seed for interpolation", default=9876543210),
        interpolation_steps: int = Input(description="Number of interpolation steps between seeds", default=4),
        interpolation_strength: float = Input(description="How far to interpolate towards the second seed (0.05 = 5%)", default=0.05),
        create_video: bool = Input(description="Create a video from the interpolated images", default=True),
        fps: int = Input(description="Frames per second for the video", default=5),
        output_format: str = Input(
            description="Format of the output image",
            choices=["webp", "jpg", "png"],
            default="png",
        ),
    ) -> Union[Path, List[Path]]:
        print(f"Using seed1: {seed1}")
        print(f"Using seed2: {seed2}")
        print(f"Interpolation steps: {interpolation_steps}")
        print(f"Interpolation strength: {interpolation_strength * 100}%")
        print(f"Prompt: {prompt}")
        print(f"Dimensions: {width}x{height}")
        
        # Make sure dimensions are multiples of 16
        width = self.make_multiple_of_16(width)
        height = self.make_multiple_of_16(height)
        
        # Calculate latent dimensions (1/8 of the image dimensions)
        latent_height = height // 8
        latent_width = width // 8
        
        # Set up random generators
        generator1 = torch.Generator(device="cuda").manual_seed(seed1)
        generator2 = torch.Generator(device="cuda").manual_seed(seed2)
        
        # Directly create latent vectors with the known shape [1, 16, 96, 96]
        latent1 = torch.randn(
            (1, 16, latent_height, latent_width),
            generator=generator1,
            device="cuda"
        )
        
        latent2 = torch.randn(
            (1, 16, latent_height, latent_width),
            generator=generator2,
            device="cuda"
        )
        
        print(f"Created latent1 shape: {latent1.shape}")
        print(f"Created latent2 shape: {latent2.shape}")
        
        # Initialize list to store all generated images
        all_images = []
        
        # Generate images for each interpolation step (including endpoints)
        total_steps = interpolation_steps
        
        for i, t in enumerate(np.linspace(0, interpolation_strength, total_steps + 1)):
            # For t=0, use latent1 directly
            if t == 0:
                current_latent = latent1
            else:
                # Interpolate between the latents
                # Reshape latents to 2D for SLERP
                latent1_flat = latent1.reshape(1, -1)
                latent2_flat = latent2.reshape(1, -1)
                
                # Apply SLERP
                interpolated_latent_flat = slerp(t, latent1_flat, latent2_flat)
                
                # Reshape back to original shape
                current_latent = interpolated_latent_flat.reshape(latent1.shape)
            
            print(f"Generating image {i+1}/{total_steps+1} with t={t}")
            
            # Log detailed shape information
            print(f"Current latent shape: {current_latent.shape}")
            print(f"Current latent dtype: {current_latent.dtype}")
            print(f"Current latent device: {current_latent.device}")
            
            # Pack the latents before passing to the transformer
            # Step 1: View as a 6D tensor with spatially split dimensions
            packed_latent = current_latent.view(1, 16, latent_height // 2, 2, latent_width // 2, 2)
            # Step 2: Permute to place spatial dimensions together
            packed_latent = packed_latent.permute(0, 2, 4, 1, 3, 5)
            # Step 3: Reshape to final packed form [batch, tokens, channels]
            packed_latent = packed_latent.reshape(1, (latent_height // 2) * (latent_width // 2), 16 * 4)
            
            print(f"Packed latent shape: {packed_latent.shape}")
            
            # Generate image with the current latent
            output = self.pipe(
                prompt=prompt,
                guidance_scale=0.0,
                width=width,
                height=height,
                num_inference_steps=4,
                max_sequence_length=256,
                latents=packed_latent,
                output_type="pil"
            )
            
            # Log tensor shapes captured during forward pass
            print("Tensor shapes during forward pass:")
            for key, shape in self.tensor_shapes.items():
                print(f"  {key}: {shape}")
            
            all_images.append(output.images[0])
        
        # Create output paths for the images
        image_paths = []
        # Create output directory if it doesn't exist
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        for i, img in enumerate(all_images):
            image_path = f"{output_dir}/interpolated_{i}.{output_format}"
            img.save(image_path)
            image_paths.append(image_path)
        
        # Create video if requested
        if create_video and len(image_paths) > 1:
            video_path = f"{output_dir}/interpolation_video.mp4"
            create_video_from_images(image_paths, video_path, fps=fps)
            return Path(video_path)
        
        # Return the first image if only one was generated
        if len(image_paths) == 1:
            return Path(image_paths[0])
        
        # Return all images as a list
        return [Path(p) for p in image_paths]

def create_video_from_images(image_paths, output_path, fps=10):
    """Create a video from a list of image paths using ffmpeg"""
    # Create a temporary directory for frame numbering
    tmp_dir = '/tmp/frames'
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Copy and rename images with sequential numbering
    for i, img_path in enumerate(image_paths):
        # Create a symbolic link with sequential numbering
        output_frame = f"{tmp_dir}/frame_{i:04d}.{img_path.split('.')[-1]}"
        # Copy the file instead of creating a symlink
        subprocess.run(['cp', img_path, output_frame], check=True)
    
    # Use ffmpeg to create the video from the sequentially numbered frames
    cmd = [
        'ffmpeg', 
        '-y',  # Overwrite output file if it exists
        '-framerate', str(fps),
        '-i', f"{tmp_dir}/frame_%04d.{image_paths[0].split('.')[-1]}", 
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path
    ]
    subprocess.run(cmd, check=True)
    
    return output_path




# Function for spherical interpolation (SLERP)
def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """
    Spherical linear interpolation
    Args:
        t: Float value between 0.0 and 1.0
        v0: Starting vector
        v1: Final vector
        DOT_THRESHOLD: Threshold for linear interpolation fallback
    """
    # v0 and v1 should be normalized vectors
    if not isinstance(v0, torch.Tensor):
        v0 = torch.tensor(v0).float()
    if not isinstance(v1, torch.Tensor):
        v1 = torch.tensor(v1).float()
    
    # Copy the vectors to reuse them later
    v0_copy = v0.clone().detach()
    v1_copy = v1.clone().detach()
    
    # Normalize the vectors to get the directions and angles
    v0 = v0 / torch.norm(v0)
    v1 = v1 / torch.norm(v1)
    
    # Dot product with clamp to stay in range
    dot = torch.sum(v0 * v1)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # If the inputs are too close for comfort, linearly interpolate
    if abs(dot) > DOT_THRESHOLD:
        return v0_copy + t * (v1_copy - v0_copy)
    
    # Calculate initial angle between v0 and v1
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    
    # Finish the slerp algorithm
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    
    return s0 * v0_copy + s1 * v1_copy
