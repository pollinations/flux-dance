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
import librosa

MODEL_CACHE = "FLUX.1-schnell"
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/files.tar"
HALF_PRECISION_CACHE = "FLUX.1-schnell-fp16"  # New location for half precision model
# INT8_CACHE = "FLUX.1-schnell-int8"  # New INT8 quantized model path (commented out)

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

def get_amplitude_envelope(signal, hop_length):
    """Calculate the amplitude envelope of a signal with a given frame size and hop length."""
    amplitude_envelope = []
    
    # Calculate amplitude envelope for each frame
    for i in range(0, len(signal), hop_length): 
        current_frame = signal[i:i+hop_length]
        if len(current_frame) > 0:  # Ensure the frame has data
            amplitude_envelope_current_frame = max(np.abs(current_frame))
            amplitude_envelope.append(amplitude_envelope_current_frame)
    
    return np.array(amplitude_envelope)

def process_audio(audio_file, frame_rate=30, smoothing=0.8, loudness_type="peak"):
    """Process audio file to extract amplitude information for interpolation."""
    print(f"Processing audio file: {audio_file}")
    
    # Load audio file
    y, sr = librosa.load(str(audio_file), sr=22050)
    
    # Calculate hop length based on frame rate
    hop_length = int(sr / frame_rate)
    print(f"Audio sample rate: {sr}, hop length: {hop_length}")
    
    # Get audio intensities based on selected method
    if loudness_type == "peak":
        # Get amplitude envelope
        intensities = get_amplitude_envelope(y, hop_length)
    else:  # "rms"
        # Get RMS (root mean square) values
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)
        intensities = rms[0]
    
    # Normalize intensities to 0-1 range
    if len(intensities) > 0:
        intensities = intensities / intensities.max()
    
    # Apply smoothing
    smoothed_intensities = []
    current_smoothed = 0
    for intensity in intensities:
        current_smoothed = (current_smoothed * smoothing) + (intensity * (1 - smoothing))
        smoothed_intensities.append(current_smoothed)
    
    return np.array(smoothed_intensities)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        start_time = time.time()
        
        # Check for models in order of preference: FP16, original
        if os.path.exists(HALF_PRECISION_CACHE):
            print(f"Loading half precision model from {HALF_PRECISION_CACHE}")
            model_path = HALF_PRECISION_CACHE
        else:
            # Download original model if needed
            if not os.path.exists(MODEL_CACHE):
                download_weights(MODEL_URL, MODEL_CACHE)
            
            # Use original model path
            print(f"Using original model from {MODEL_CACHE}")
            model_path = MODEL_CACHE
        
        # Enable TensorFloat-32 for faster computation on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
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
        
        print(f"setup took: ", time.time() - start_time)

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
        interpolation_steps: int = Input(
            description="Number of interpolation steps between seeds (only used when audio_file is not provided)",
            default=4
        ),
        interpolation_strength: float = Input(
            description="How far to interpolate towards the second seed (0.05 = 5%)",
            default=0.05
        ),
        audio_file: Path = Input(
            description="Audio file for reactive interpolation. When provided, this overrides interpolation_steps.",
            default=None
        ),
        audio_frame_rate: int = Input(
            description="Frames per second to extract from audio",
            default=30
        ),
        audio_smoothing: float = Input(
            description="Smoothing factor for audio amplitude (0-1, higher values = smoother transitions)",
            default=0.8
        ),
        audio_noise_scale: float = Input(
            description="Scale factor for audio-based interpolation (higher values = stronger effect)",
            default=0.3
        ),
        audio_loudness_type: str = Input(
            description="Method to calculate audio intensity",
            choices=["peak", "rms"],
            default="peak"
        ),
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
        
        if audio_file:
            print(f"Using audio file: {audio_file}")
            print(f"Audio frame rate: {audio_frame_rate}")
            print(f"Audio smoothing: {audio_smoothing}")
            print(f"Audio noise scale: {audio_noise_scale}")
            print(f"Audio loudness type: {audio_loudness_type}")
        else:
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
        
        # Directly create latent vectors with the known shape [1, 16, latent_height, latent_width]
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
        
        # Process audio or create interpolation steps
        if audio_file:
            # Process audio file to get intensity values
            audio_intensities = process_audio(
                audio_file,
                frame_rate=audio_frame_rate,
                smoothing=audio_smoothing,
                loudness_type=audio_loudness_type
            )
            
            # Ensure we have at least one intensity value
            if len(audio_intensities) == 0:
                audio_intensities = np.array([0.0])
                
            print(f"Extracted {len(audio_intensities)} audio intensity values")
            
            # Scale the intensities by the noise scale factor
            interpolation_values = audio_intensities * audio_noise_scale
            
            # Cap at the maximum interpolation strength
            interpolation_values = np.clip(interpolation_values, 0, interpolation_strength)
            
            total_steps = len(interpolation_values)
        else:
            # Use linear interpolation steps if no audio file
            interpolation_values = np.linspace(0, interpolation_strength, interpolation_steps + 1)
            total_steps = interpolation_steps + 1
        
        # Generate images for each interpolation value
        for i, t in enumerate(interpolation_values):
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
            
            print(f"Generating image {i+1}/{total_steps} with t={t}")
            
            # Log detailed shape information
            print(f"Current latent shape: {current_latent.shape}")
            
            # Pack the latents before passing to the transformer
            # Step 1: View as a 6D tensor with spatially split dimensions
            packed_latent = current_latent.view(1, 16, latent_height // 2, 2, latent_width // 2, 2)
            # Step 2: Permute to place spatial dimensions together
            packed_latent = packed_latent.permute(0, 2, 4, 1, 3, 5)
            # Step 3: Reshape to final packed form [batch, tokens, channels]
            packed_latent = packed_latent.reshape(1, (latent_height // 2) * (latent_width // 2), 16 * 4)
            
            print(f"Packed latent shape: {packed_latent.shape}")
            
            # Generate image with the current latent
            with torch.no_grad():
                outputs = self.pipe(
                    prompt=prompt,
                    latents=packed_latent,
                    negative_prompt="blurry, ugly, deformed, out of frame",
                )
            
            # Log tensor shapes if available
            if self.tensor_shapes:
                print("Tensor shapes:")
                for name, shape in self.tensor_shapes.items():
                    print(f"  {name}: {shape}")
            
            # Get the generated image
            image = outputs.images[0]
            
            # Create output directory if it doesn't exist
            os.makedirs("outputs", exist_ok=True)
            
            # Save the image with the current timestamp and step
            timestamp = int(time.time())
            image_filename = f"outputs/output_{timestamp}_{i:04d}.{output_format}"
            image.save(image_filename)
            
            # Add to the list of generated images
            all_images.append(Path(image_filename))
        
        # Create a video if requested
        if create_video and len(all_images) > 1:
            video_path = f"outputs/output_{timestamp}_video.mp4"
            create_video_from_images(all_images, video_path, fps)
            
            # Return the video path
            return Path(video_path)
        
        # Return all generated images
        return all_images

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
