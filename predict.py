# Prediction interface for Cog 
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
import numpy as np
from typing import List, Union
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderKL, GGUFQuantizationConfig
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from optimum.quanto import freeze, qfloat8, quantize
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
import shutil
import tempfile
import librosa

MODEL_CACHE = "FLUX.1-schnell"
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/files.tar"
HALF_PRECISION_CACHE = "FLUX.1-schnell-fp16"  # Location for half precision model
GGUF_MODEL_ID = "city96/FLUX.1-schnell-gguf"  # HuggingFace GGUF model ID
GGUF_MODEL_FILE = "flux1-schnell-Q6_K.gguf"   # GGUF model filename
BFL_REPO = "black-forest-labs/FLUX.1-schnell" # Original model repo
REVISION = "refs/pr/1"                        # Revision for the original model

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

def download_hf_model(model_id, local_path):
    """Download model from HuggingFace to local path"""
    start = time.time()
    print(f"Downloading model {model_id} to {local_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)
    
    # Download the model files
    snapshot_download(
        repo_id=model_id,
        local_dir=local_path,
        local_dir_use_symlinks=False
    )
    
    print(f"Download completed in {time.time() - start:.2f} seconds")
    return os.path.exists(local_path)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        start_time = time.time()
        
        # Enable TensorFloat-32 for faster computation on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Set memory optimization settings
        torch.cuda.empty_cache()
        
        # Try to load the GGUF model first
        try:
            print(f"Attempting to load GGUF model from {GGUF_MODEL_ID}")
            self.load_gguf_model()
        except Exception as e:
            print(f"Failed to load GGUF model: {e}")
            
            # Fall back to half precision model if available
            if os.path.exists(HALF_PRECISION_CACHE):
                print(f"Loading half precision model from {HALF_PRECISION_CACHE}")
                self.load_standard_model(HALF_PRECISION_CACHE)
            else:
                # Download original model if needed
                if not os.path.exists(MODEL_CACHE):
                    download_weights(MODEL_URL, MODEL_CACHE)
                
                # Use original model path
                print(f"Using original model from {MODEL_CACHE}")
                self.load_standard_model(MODEL_CACHE)
        
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
    
    def load_standard_model(self, model_path):
        """Load the standard model using diffusers pipeline"""
        print(f"Loading Flux.schnell Pipeline from {model_path}")
        self.pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use half precision
            low_cpu_mem_usage=True,     # Optimize CPU memory usage during loading
        )
        
        # Move to CUDA in a memory-efficient way
        print("Moving model to CUDA...")
        self.pipe.to("cuda")
    
    def load_gguf_model(self):
        """Load the GGUF quantized model from HuggingFace"""
        print("Loading GGUF quantized model...")
        
        # Download the GGUF model file if needed
        gguf_path = hf_hub_download(
            repo_id=GGUF_MODEL_ID,
            filename=GGUF_MODEL_FILE
        )
        
        print(f"GGUF model downloaded to: {gguf_path}")
        
        # Load the transformer with GGUF quantization
        transformer = FluxTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16
        )
        
        # Load the rest of the pipeline components
        self.pipe = FluxPipeline.from_pretrained(
            BFL_REPO,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            revision=REVISION
        )
        
        # Enable model CPU offload to save GPU memory
        self.pipe.enable_model_cpu_offload()
        print("GGUF model loaded successfully")
    
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
        
        # Get amplitude measurements based on loudness type
        if loudness_type == "peak":
            # Use numpy's efficient array operations instead of loops
            # Reshape audio into frames
            frames = librosa.util.frame(y, frame_length=hop_length, hop_length=hop_length).T
            # Get maximum absolute value for each frame (peak amplitude)
            amplitudes = np.max(np.abs(frames), axis=1)
        else:
            # Use librosa's built-in RMS function (much faster)
            amplitudes = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        
        # Normalize in one step
        if np.max(amplitudes) > 0:
            normalized_amplitudes = amplitudes / np.max(amplitudes)
        else:
            normalized_amplitudes = np.zeros_like(amplitudes)
        
        # Apply smoothing using vectorized operations
        if smoothing > 0:
            # Use exponential moving average for smoothing
            smoothed = np.zeros_like(normalized_amplitudes)
            smoothed[0] = normalized_amplitudes[0]
            for i in range(1, len(normalized_amplitudes)):
                smoothed[i] = smoothing * smoothed[i-1] + (1 - smoothing) * normalized_amplitudes[i]
            
            # Apply noise scale
            return smoothed * noise_scale
        else:
            # If no smoothing, just apply noise scale directly
            return normalized_amplitudes * noise_scale

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
        
        print(f"Created latent1 with seed {seed1}")#, shape: {latent1.shape}")
        print(f"Created latent2 with seed {seed2}")#, shape: {latent2.shape}")
        
        # Initialize list to store all generated images
        all_images = []
        
        # Determine interpolation values based on audio file or steps
        if audio_file is not None:
            # Get audio amplitudes to drive interpolation
            print(f"Using audio file {audio_file} for interpolation. Getting amplitudes...")
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
            
        # Pre-compute prompt embeddings for each unique prompt
        unique_prompts = list(set(prompt_for_frame))
        print(f"Pre-computing embeddings for {len(unique_prompts)} unique prompts...")
        prompt_embeddings = {}
        
        for unique_prompt in unique_prompts:
            # # Tokenize the prompt
            # text_inputs = self.pipe.tokenizer(
            #     unique_prompt,
            #     padding="max_length",
            #     max_length=self.pipe.tokenizer.model_max_length,
            #     max_sequence_length=256,
            #     truncation=True,
            #     return_tensors="pt",
            # ).to("cuda")
            
            # # Generate the prompt embeddings
            # with torch.no_grad():
            #     prompt_embeds = self.pipe.text_encoder(
            #         text_inputs.input_ids,
            #         attention_mask=text_inputs.attention_mask,
            #     ).last_hidden_state
            
            # # Store the embeddings
            # prompt_embeddings[unique_prompt] = prompt_embeds
            
            prompt_embeds, pooled_prompt_embeds, _text_ids = self.pipe.encode_prompt(
                prompt=unique_prompt, prompt_2=None, max_sequence_length=256
            )
            
            # Store the embeddings
            prompt_embeddings[unique_prompt] = (prompt_embeds, pooled_prompt_embeds)

        print(f"Finished pre-computing prompt embeddings")
            
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
            
            print("Packing latents...")
            # Pack the latents before passing to the transformer
            # Step 1: View as a 6D tensor with spatially split dimensions
            packed_latent = current_latent.view(1, 16, latent_height // 2, 2, latent_width // 2, 2)
            # Step 2: Permute to place spatial dimensions together
            packed_latent = packed_latent.permute(0, 2, 4, 1, 3, 5)
            # Step 3: Reshape to final packed form [batch, tokens, channels]
            packed_latent = packed_latent.reshape(1, (latent_height // 2) * (latent_width // 2), 16 * 4)
            
            print(f"Packed latent")
            
            # Use the pre-computed prompt embeddings instead of passing the prompt text
            output = self.pipe(
                prompt_embeds=prompt_embeddings[current_prompt][0],
                pooled_prompt_embeds=prompt_embeddings[current_prompt][1],
                guidance_scale=0.0,
                width=width,
                height=height,
                num_inference_steps=4,
                max_sequence_length=256,
                latents=packed_latent,
                output_type="pil"
            )
            
            all_images.append(output.images[0])
            
        return all_images

    @torch.inference_mode()
    def predict(
        self,
        style_prefix: str = Input(
            description="Style prefix to prepend to each prompt", 
            default="unsplash, crystal cubism, angular, academic art, cubism, an abstract drawing by Svetoslav Roerich, reddit contest winner, abstract illusionism, concept art, angular, apocalypse landscape broken-stained-glass, digital lines, ((woodblock)), concrete poetry, anton semono. black and white, granular, abstract, hand drawings, old, grainy, retro, art album, fade, drawing on paper, stone texture, wood texture, lava texture, mysticism, sacred symbology, blueprint, ancient document, buddhism, erosion, decay, prism, rage, fade."),
        narrative: str = Input(
            description="Narrative prompts, one per line. Each line defines a prompt for a certain time in the sequence.", 
            default="""granular Silhouettes of valleys 
Granular Silhouettes of hills"""),
        audio_file: Path = Input(description="Audio file to drive interpolation (optional)", default=None),
        width: int = Input(description="Width of the generated image", default=512),
        height: int = Input(description="Height of the generated image", default=512),
        seed: int = Input(description="Random seed for interpolation (seed+1 will be used as the second seed)", default=42),
        interpolation_steps: int = Input(description="Number of interpolation steps between seeds (ignored if audio_file is provided)", default=4),
        variation_strength: float = Input(description="Controls the amount of variation between frames (0-1)", default=0.3),
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
                raise ValueError("Narrative cannot be empty")
            print(f"Using narrative with {len(prompts)} prompts")
        else:
            raise ValueError("Narrative is required")
        
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
