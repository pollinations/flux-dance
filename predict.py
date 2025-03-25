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
import shutil
import tempfile
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

# def convert_to_int8():
#     """Convert the model to INT8 quantization using the conversion script"""
#     print("Converting model to INT8 quantization...")
#     try:
#         # Check if the conversion script exists
#         if not os.path.exists("convert_model_int8.py"):
#             print("INT8 conversion script not found, skipping conversion")
#             return False
            
#         # Run the conversion script
#         result = subprocess.run(
#             ["python3", "convert_model_int8.py"], 
#             capture_output=True, 
#             text=True,
#             check=True
#         )
#         print(result.stdout)
#         return os.path.exists(INT8_CACHE)
#     except subprocess.CalledProcessError as e:
#         print(f"Error converting model to INT8: {e}")
#         print(f"Error output: {e.stderr}")
#         return False

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

    @staticmethod
    def get_audio_amplitudes(audio_file, frame_rate=10, smoothing=0.8, noise_scale=0.3, loudness_type="peak"):
        """Extract amplitude envelope from an audio file.
        
        Args:
            audio_file: Path to the audio file
            frame_rate: Number of frames per second to extract
            smoothing: Smoothing factor (0-1), higher values mean more smoothing
            noise_scale: Scale factor for the audio amplitudes (same as variation_strength)
            loudness_type: Type of loudness measurement ('peak' or 'rms')
            
        Returns:
            Normalized amplitude values
        """
        # Load audio file
        y, sr = librosa.load(audio_file, sr=22050)
        
        # Calculate hop length based on frame rate
        hop_length = int(22050 / frame_rate)
        
        # Initialize smoothed intensity
        smoothed_intensity = 0
        smoothed_intensities = []
        
        if loudness_type == "peak":
            # Get amplitude envelope
            amplitude_envelope = []
            
            # Calculate amplitude envelope for each frame
            for i in range(0, len(y), hop_length):
                current_frame = np.abs(y[i:i+hop_length])
                if len(current_frame) > 0:
                    amplitude_envelope_current_frame = np.max(current_frame)
                    amplitude_envelope.append(amplitude_envelope_current_frame)
            
            # Convert to numpy array
            amplitude_envelope = np.array(amplitude_envelope)
            
            # Normalize
            if amplitude_envelope.max() > 0:
                normalized_amplitudes = amplitude_envelope / amplitude_envelope.max()
            else:
                normalized_amplitudes = np.zeros_like(amplitude_envelope)
        else:
            # Get RMS (root mean square) - energy
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)
            
            # Normalize
            if rms[0].max() > 0:
                normalized_amplitudes = rms[0] / rms[0].max()
            else:
                normalized_amplitudes = np.zeros_like(rms[0])
        
        # Apply smoothing and scaling
        for amplitude in normalized_amplitudes:
            smoothed_intensity = (smoothed_intensity * smoothing) + (amplitude * (1 - smoothing))
            smoothed_intensities.append(smoothed_intensity * noise_scale)
        
        return np.array(smoothed_intensities)

    @torch.inference_mode()
    def generate_interpolated_images(
        self,
        prompts: Union[str, List[str]],
        width: int,
        height: int,
        seed: int,
        interpolation_steps: int = None,
        variation_strength: float = None,
        audio_file: Path = None,
        audio_smoothing: float = 0.8,
        audio_loudness_type: str = "peak",
        fps: int = 10,
        style_prefix: str = "",
    ) -> List[Image.Image]:
        """Generate a series of interpolated images between two random seeds.
        
        Args:
            prompts: Text prompt(s) for image generation (single string or list of strings)
            width: Width of the generated image
            height: Height of the generated image
            seed: Random seed for interpolation (seed+1 will be used as the second seed)
            interpolation_steps: Number of interpolation steps between seeds (ignored if audio_file is provided)
            variation_strength: Controls the amount of variation between frames (0-1)
            audio_file: Path to audio file for audio-driven interpolation
            audio_smoothing: Smoothing factor for audio amplitudes
            audio_loudness_type: Type of loudness measurement ('peak' or 'rms')
            fps: Frames per second for audio-driven interpolation
            style_prefix: Style prefix to prepend to each prompt
            
        Returns:
            List of PIL images generated through interpolation
        """
        # Make sure dimensions are multiples of 16
        width = self.make_multiple_of_16(width)
        height = self.make_multiple_of_16(height)
        
        # Calculate latent dimensions (1/8 of the image dimensions)
        latent_height = height // 8
        latent_width = width // 8
        
        # Set up random generators with seed and seed+1
        seed1 = seed
        seed2 = seed + 1
        
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
        
        print(f"Created latent1 with seed {seed1}, shape: {latent1.shape}")
        print(f"Created latent2 with seed {seed2}, shape: {latent2.shape}")
        
        # Initialize list to store all generated images
        all_images = []
        
        # Determine interpolation values based on audio file or steps
        if audio_file is not None:
            # Get audio amplitudes to drive interpolation
            interpolation_values = self.get_audio_amplitudes(
                audio_file, 
                frame_rate=fps,
                smoothing=audio_smoothing,
                noise_scale=variation_strength,  # Use variation_strength instead of audio_noise_scale
                loudness_type=audio_loudness_type
            )
            total_steps = len(interpolation_values)
            print(f"Using audio file for interpolation with {total_steps} frames")
        else:
            # Use linear interpolation with fixed steps
            total_steps = interpolation_steps
            interpolation_values = np.linspace(0, variation_strength, total_steps + 1)
            print(f"Using fixed interpolation with {total_steps} steps")
        
        # Handle multiple prompts if provided
        if isinstance(prompts, list):
            # If we have multiple prompts, we need to determine which prompt to use for each frame
            if len(prompts) == 1:
                # If only one prompt, use it for all frames
                prompt_for_frame = [prompts[0]] * len(interpolation_values)
            else:
                # If we have multiple prompts, distribute them across the frames
                prompt_for_frame = []
                num_frames_per_prompt = len(interpolation_values) // (len(prompts) - 1)
                
                # Handle edge case where we have more prompts than frames
                if num_frames_per_prompt == 0:
                    num_frames_per_prompt = 1
                    # Truncate prompts list if needed
                    prompts = prompts[:len(interpolation_values) + 1]
                
                # For each pair of consecutive prompts
                for i in range(len(prompts) - 1):
                    # Calculate the number of frames for this prompt pair
                    if i == len(prompts) - 2:
                        # Last pair gets all remaining frames
                        frames_for_pair = len(interpolation_values) - len(prompt_for_frame)
                    else:
                        frames_for_pair = num_frames_per_prompt
                    
                    # Distribute frames between current prompt and next prompt
                    for j in range(frames_for_pair):
                        # Calculate interpolation factor between prompts
                        t_prompt = j / frames_for_pair
                        # For simplicity, we're not interpolating between prompts,
                        # just using the first prompt for the first half and second for the second half
                        if t_prompt < 0.5:
                            prompt_for_frame.append(prompts[i])
                        else:
                            prompt_for_frame.append(prompts[i + 1])
        else:
            # Single prompt for all frames
            prompt_for_frame = [prompts] * len(interpolation_values)
        
        # Apply style prefix if provided
        if style_prefix:
            prompt_for_frame = [f"{style_prefix} {prompt}" for prompt in prompt_for_frame]
            
        # Generate images for each interpolation step
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
            
            current_prompt = prompt_for_frame[i]
            print(f"Generating image {i+1}/{len(interpolation_values)} with t={t}, prompt: {current_prompt}")
            
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
                prompt=current_prompt,
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
            
        return all_images

    @torch.inference_mode()
    def predict(
        self,
        style_prefix: str = Input(
            description="Style prefix to prepend to each prompt", 
            default="A painting by paul klee, intricate details of"),
        prompt: str = Input(description="Prompt for generated image", default="A beautiful landscape"),
        narrative: str = Input(
            description="Narrative prompts, one per line. Each line defines a prompt for a certain time in the sequence.", 
            default="""A painting of a moth
A painting of a killer dragonfly
Two fishes talking to eachother in deep sea"""),
        width: int = Input(description="Width of the generated image", default=1024),
        height: int = Input(description="Height of the generated image", default=1024),
        seed: int = Input(description="Random seed for interpolation (seed+1 will be used as the second seed)", default=42),
        interpolation_steps: int = Input(description="Number of interpolation steps between seeds (ignored if audio_file is provided)", default=4),
        variation_strength: float = Input(description="Controls the amount of variation between frames (0-1)", default=0.3),
        audio_file: Path = Input(description="Audio file to drive interpolation (optional)", default=None),
        audio_smoothing: float = Input(description="Smoothing factor for audio (0-1, higher = more smoothing)", default=0.8),
        audio_loudness_type: str = Input(
            description="Type of audio loudness measurement to use",
            choices=["peak", "rms"],
            default="peak"
        ),
        create_video: bool = Input(description="Create a video from the interpolated images", default=True),
        fps: int = Input(description="Frames per second for the video", default=10),
        output_format: str = Input(
            description="Format of the output image",
            choices=["webp", "jpg", "png"],
            default="png",
        ),
    ) -> Path:
        print(f"Using seed: {seed} (and {seed+1} as second seed)")
        
        if audio_file:
            print(f"Using audio file: {audio_file}")
            print(f"Audio smoothing: {audio_smoothing}")
            print(f"Variation strength: {variation_strength}")
            print(f"Audio loudness type: {audio_loudness_type}")
            print(f"FPS: {fps}")
        else:
            print(f"Interpolation steps: {interpolation_steps}")
            print(f"Variation strength: {variation_strength * 100}%")
        
        # Process prompts
        if narrative:
            # Split narrative into individual prompts
            prompts = [p.strip() for p in narrative.strip().split('\n') if p.strip()]
            if len(prompts) == 0:
                # If narrative is empty or invalid, fall back to single prompt
                prompts = [prompt]
            print(f"Using narrative with {len(prompts)} prompts")
        else:
            # Use single prompt
            prompts = [prompt]
            print(f"Using single prompt: {prompt}")
        
        if style_prefix:
            print(f"Applying style prefix to all prompts: {style_prefix}")
        
        print(f"Dimensions: {width}x{height}")
        
        # Create a temporary directory for the output
        with tempfile.TemporaryDirectory() as output_dir:
            # Generate interpolated images
            images = self.generate_interpolated_images(
                prompts=prompts,
                width=width,
                height=height,
                seed=seed,
                interpolation_steps=interpolation_steps,
                variation_strength=variation_strength,
                audio_file=audio_file,
                audio_smoothing=audio_smoothing,
                audio_loudness_type=audio_loudness_type,
                fps=fps,
                style_prefix=style_prefix,
            )
            
            # Save images to temporary directory
            image_paths = []
            for i, img in enumerate(images):
                image_path = f"{output_dir}/interpolation_{i:04d}.{output_format}"
                img.save(image_path)
                image_paths.append(image_path)
            
            # Create video if requested
            if create_video and len(image_paths) > 1:
                # Create video in the temporary directory
                video_path = f"{output_dir}/interpolation_video.mp4"
                
                if audio_file and os.path.exists(audio_file):
                    # Create video with audio
                    create_video_from_images(image_paths, video_path, fps=fps, audio_file=audio_file)
                else:
                    # Create video without audio
                    create_video_from_images(image_paths, video_path, fps=fps)
                
                # Create a path in the Cog-managed directory for the final output
                final_video_path = Path(f"/tmp/output_video_{int(time.time())}.mp4")
                shutil.copy2(video_path, final_video_path)
                
                # Return the copied video file
                return final_video_path
            
            # If no video was created, return the first image or all images
            if len(image_paths) == 1:
                final_image_path = Path(f"/tmp/output_image_{int(time.time())}.{output_format}")
                shutil.copy2(image_paths[0], final_image_path)
                return final_image_path
            
            # Return all images as a list (copying them to persistent storage first)
            final_image_paths = []
            for i, img_path in enumerate(image_paths):
                final_path = Path(f"/tmp/output_image_{int(time.time())}_{i}.{output_format}")
                shutil.copy2(img_path, final_path)
                final_image_paths.append(final_path)
            
            return final_image_paths

def create_video_from_images(image_paths, output_path, fps=10, audio_file=None):
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
        '-i', f"{tmp_dir}/frame_%04d.{image_paths[0].split('.')[-1]}"
    ]
    
    # Add audio input if provided
    if audio_file and os.path.exists(audio_file):
        cmd.extend(['-i', str(audio_file), '-map', '0:v', '-map', '1:a', '-shortest'])
    
    # Add encoding options
    cmd.extend([
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart'
    ])
    
    # Add output path
    cmd.append(output_path)
    
    # Run ffmpeg command
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
