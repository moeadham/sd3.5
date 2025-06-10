import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Track total execution time
total_execution_start = time.time()

# Track total import time
total_import_start = time.time()

# Time each import
logger.info("Starting imports...")

import_start = time.time()
import numpy as np
import_time = time.time() - import_start
logger.info(f"numpy import took {import_time:.4f} seconds")

import_start = time.time()
import random
import_time = time.time() - import_start
logger.info(f"random import took {import_time:.4f} seconds")

import_start = time.time()
import os
import_time = time.time() - import_start
logger.info(f"os import took {import_time:.4f} seconds")

import_start = time.time()
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from transformers import CLIPTextModelWithProjection, T5EncoderModel
from diffusers import AutoencoderKL, SD3Transformer2DModel
import diffusers
import transformers
transformers.utils.logging.set_verbosity_info()
diffusers.utils.logging.set_verbosity_debug()
import_time = time.time() - import_start
logger.info(f"diffusers import took {import_time:.4f} seconds")

import_start = time.time()
from diffusers.utils import load_image
import_time = time.time() - import_start
logger.info(f"diffusers.utils import took {import_time:.4f} seconds")

import_start = time.time()
import torch
import_time = time.time() - import_start
logger.info(f"torch import took {import_time:.4f} seconds")

import_start = time.time()
from huggingface_hub import login
import_time = time.time() - import_start
logger.info(f"huggingface_hub import took {import_time:.4f} seconds")

import_start = time.time()
import cv2
import_time = time.time() - import_start
logger.info(f"cv2 import took {import_time:.4f} seconds")

import_start = time.time()
from PIL import Image
import_time = time.time() - import_start
logger.info(f"PIL import took {import_time:.4f} seconds")

import_start = time.time()
from transformers import DPTImageProcessor, DPTForDepthEstimation
import_time = time.time() - import_start
logger.info(f"transformers import took {import_time:.4f} seconds")

total_import_time = time.time() - total_import_start
logger.info(f"Total import time: {total_import_time:.4f} seconds")

# Set cache directories
os.environ["HF_HOME"] = "./cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "hub"
os.environ["TRANSFORMERS_CACHE"] = "transformers"
os.environ["HF_DATASETS_CACHE"] = "datasets"
os.environ["HF_HUB_OFFLINE"] = "1"  # Force offline mode

#logger.info("Starting HuggingFace login...")
#login_start = time.time()
#login(os.getenv("HF_TOKEN"))
#login_time = time.time() - login_start
#logger.info(f"HuggingFace login took {login_time:.4f} seconds")

device = "cuda"
logger.info(f"Using device: {device}")

# Set cache directory
cache_dir = "./cache"
model_repo_id = "stabilityai/stable-diffusion-3.5-large"
## need to convert to diffusers format. 
## https://github.com/huggingface/diffusers/blob/6c7fad7ec8b2417c92326804e1751658874fd43b/scripts/convert_sd3_controlnet_to_diffusers.py#L2
#python scripts/convert_sd3_controlnet_to_diffusers.py --checkpoint_path "../sd3.5/models/sd3.5_large_controlnet_depth.safetensors" --output_path ../sd3.5/models/sd3.5_large_controlnet_depth_diffusers
controlnet_repo_id = "/workspace/sd3.5/models/sd3.5_large_controlnet_depth_diffusers"
torch_dtype = torch.bfloat16

logger.info(f"Model repo: {model_repo_id}")
logger.info(f"ControlNet repo: {controlnet_repo_id}")
logger.info(f"Cache directory: {cache_dir}")
logger.info(f"Torch dtype: {torch_dtype}")

# === DEPTH PROCESSING PHASE ===
depth_processing_start = time.time()

logger.info("Loading Depth estimator...")
depth_estimator_load_start = time.time()
depth_estimator = DPTForDepthEstimation.from_pretrained(
    "Intel/dpt-hybrid-midas",
    cache_dir=cache_dir,
    torch_dtype=torch_dtype,
    local_files_only=True,
).to("cuda")
depth_estimator_load_time = time.time() - depth_estimator_load_start
logger.info(f"Depth estimator loading took {depth_estimator_load_time:.4f} seconds")

logger.info("Loading Depth feature extractor...")
feature_extractor_load_start = time.time()
feature_extractor = DPTImageProcessor.from_pretrained(
    "Intel/dpt-hybrid-midas",
    cache_dir=cache_dir,
    torch_dtype=torch_dtype,
    local_files_only=True
)
feature_extractor_load_time = time.time() - feature_extractor_load_start
logger.info(f"Depth feature extractor loading took {feature_extractor_load_time:.4f} seconds")

# download an image
logger.info("Loading input image...")
image_load_start = time.time()
image = load_image(
    "./inputs/square.png"
)
image_load_time = time.time() - image_load_start
logger.info(f"Input image loading took {image_load_time:.4f} seconds")

logger.info("Generating depth map...")
depth_generation_start = time.time()
depth_image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
with torch.no_grad(), torch.autocast("cuda"):
    depth_map = depth_estimator(depth_image).predicted_depth

depth_map = torch.nn.functional.interpolate(
    depth_map.unsqueeze(1),
    size=(1024, 1024),
    mode="bicubic",
    align_corners=False,
)
depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
depth_map = (depth_map - depth_min) / (depth_max - depth_min)
depth_image = torch.cat([depth_map] * 3, dim=1)
depth_image = depth_image.permute(0, 2, 3, 1).cpu().numpy()[0]
depth_image = Image.fromarray((depth_image * 255.0).clip(0, 255).astype(np.uint8))
os.makedirs("outputs/diffusers", exist_ok=True)
depth_image.save("outputs/diffusers/diffusers_depth_control.png")
depth_generation_time = time.time() - depth_generation_start
logger.info(f"Depth map generation took {depth_generation_time:.4f} seconds")

total_depth_processing_time = time.time() - depth_processing_start
logger.info(f"Total depth processing phase took {total_depth_processing_time:.4f} seconds")

# === MODEL LOADING PHASE ===
model_loading_start = time.time()

# load control net and stable diffusion v1-5
logger.info("Loading ControlNet model...")
controlnet_load_start = time.time()
controlnet = SD3ControlNetModel.from_pretrained(
    controlnet_repo_id, 
    torch_dtype=torch_dtype,
    local_files_only=True,
).to("cuda")
controlnet_load_time = time.time() - controlnet_load_start
logger.info(f"ControlNet loading took {controlnet_load_time:.4f} seconds")

########################################################
# Load VAE
logger.info("Loading VAE...")
vae_load_start = time.time()
vae = AutoencoderKL.from_pretrained(
    model_repo_id,
    subfolder="vae",
    torch_dtype=torch_dtype,
    cache_dir=cache_dir,
    local_files_only=True,
).to("cuda")
vae_load_time = time.time() - vae_load_start
logger.info(f"VAE loading took {vae_load_time:.4f} seconds")

# Load Transformer
logger.info("Loading Transformer...")
transformer_load_start = time.time()
transformer = SD3Transformer2DModel.from_pretrained(
    model_repo_id,
    subfolder="transformer",
    torch_dtype=torch_dtype,
    cache_dir=cache_dir,
    local_files_only=True,
).to("cuda")
transformer_load_time = time.time() - transformer_load_start
logger.info(f"Transformer loading took {transformer_load_time:.4f} seconds")

# Load CLIP-L text encoder
logger.info("Loading CLIP-L text encoder...")
text_encoder_load_start = time.time()
text_encoder = CLIPTextModelWithProjection.from_pretrained(
    model_repo_id,
    subfolder="text_encoder",
    torch_dtype=torch_dtype,
    cache_dir=cache_dir,
    local_files_only=True,
).to("cuda")
text_encoder_load_time = time.time() - text_encoder_load_start
logger.info(f"CLIP-L text encoder loading took {text_encoder_load_time:.4f} seconds")

# Load CLIP-G text encoder  
logger.info("Loading CLIP-G text encoder...")
text_encoder_2_load_start = time.time()
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    model_repo_id,
    subfolder="text_encoder_2",
    torch_dtype=torch_dtype,
    cache_dir=cache_dir,
    local_files_only=True,
).to("cuda")
text_encoder_2_load_time = time.time() - text_encoder_2_load_start
logger.info(f"CLIP-G text encoder loading took {text_encoder_2_load_time:.4f} seconds")

# Load T5-XXL text encoder
logger.info("Loading T5-XXL text encoder...")
text_encoder_3_load_start = time.time()
text_encoder_3 = T5EncoderModel.from_pretrained(
    #model_repo_id,
    #subfolder="text_encoder_3",
    "google/t5-large",
    torch_dtype=torch_dtype,
    cache_dir=cache_dir,
    local_files_only=True,
).to("cuda")
text_encoder_3_load_time = time.time() - text_encoder_3_load_start
logger.info(f"T5-XXL text encoder loading took {text_encoder_3_load_time:.4f} seconds")

# Load tokenizers (these are small, no GPU needed)
logger.info("Loading tokenizers...")
tokenizers_load_start = time.time()
from transformers import CLIPTokenizer, T5TokenizerFast

tokenizer = CLIPTokenizer.from_pretrained(
    model_repo_id,
    subfolder="tokenizer",
    cache_dir=cache_dir,
    local_files_only=True,
)

tokenizer_2 = CLIPTokenizer.from_pretrained(
    model_repo_id,
    subfolder="tokenizer_2",
    cache_dir=cache_dir,
    local_files_only=True,
)

tokenizer_3 = T5TokenizerFast.from_pretrained(
    model_repo_id,
    subfolder="tokenizer_3",
    cache_dir=cache_dir,
    local_files_only=True,
)
tokenizers_load_time = time.time() - tokenizers_load_start
logger.info(f"Tokenizers loading took {tokenizers_load_time:.4f} seconds")

# Load scheduler
logger.info("Loading scheduler...")
scheduler_load_start = time.time()
from diffusers import FlowMatchEulerDiscreteScheduler

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    model_repo_id,
    subfolder="scheduler",
    cache_dir=cache_dir,
    local_files_only=True,
)
scheduler_load_time = time.time() - scheduler_load_start
logger.info(f"Scheduler loading took {scheduler_load_time:.4f} seconds")

########################################################

logger.info("Assembling Stable Diffusion pipeline...")
pipeline_assembly_start = time.time()

## For 4bit quantization
# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
# model_nf4 = SD3Transformer2DModel.from_pretrained(
#     model_repo_id,
#     subfolder="transformer",
#     quantization_config=nf4_config,
#     torch_dtype=torch.bfloat16,
#     cache_dir=cache_dir,
#     local_files_only=True,
# )
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    model_repo_id, 
    controlnet=controlnet, 
    torch_dtype=torch_dtype,
    cache_dir=cache_dir,
    #transformer=model_nf4, # If you want to use 4bit quantization
    local_files_only=True,
    #low_cpu_mem_usage=True,
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    text_encoder_3=text_encoder_3,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    tokenizer_3=tokenizer_3,
    transformer=transformer,
    scheduler=scheduler,
)
# pipe.controlnet.to(torch_dtype)
# pipe.text_encoder = pipe.text_encoder.to("cuda")
# pipe.text_encoder_2 = pipe.text_encoder_2.to("cuda")
# pipe.text_encoder_3 = pipe.text_encoder_3.to("cuda")
# pipe.vae = pipe.vae.to("cuda")
pipe.to("cuda")

pipeline_assembly_time = time.time() - pipeline_assembly_start
logger.info(f"Pipeline assembly took {pipeline_assembly_time:.4f} seconds")

total_model_loading_time = time.time() - model_loading_start
logger.info(f"Total model loading phase took {total_model_loading_time:.4f} seconds")

# === GENERATION PHASE ===
logger.info("Starting image generation...")
generation_start = time.time()
#generator = torch.Generator(device="cuda").manual_seed(24)
prompt = "studio ghibli style"
image = pipe(prompt,
    negative_prompt="low quality, incomplete, blurred",
    control_image=depth_image,
    controlnet_conditioning_scale=0.85,
    #generator=generator,
    height=1024, 
    width=1024,
    num_inference_steps=60,  # SD3.5 ControlNet recommended
    guidance_scale=3.5,      # SD3.5 ControlNet recommended (lower than default)
    #control_guidance_start=0.0,
    #control_guidance_end=1.0,
).images[0]
generation_time = time.time() - generation_start
logger.info(f"Image generation took {generation_time:.4f} seconds")

logger.info("Saving generated image...")
save_start = time.time()
image.save("outputs/diffusers/diffusers_depth_output.png")
save_time = time.time() - save_start
logger.info(f"Image saving took {save_time:.4f} seconds")

# Calculate total execution time
total_execution_time = time.time() - total_execution_start

# === DETAILED TIMING BREAKDOWN ===
logger.info("=== DETAILED TIMING BREAKDOWN ===")
logger.info(f"Imports: {total_import_time:.4f}s")
logger.info(f"  - Individual model components: {depth_estimator_load_time + feature_extractor_load_time:.4f}s")
logger.info(f"  - Input processing: {image_load_time + depth_generation_time:.4f}s")
logger.info(f"  - Total depth processing: {total_depth_processing_time:.4f}s")
logger.info(f"Model loading: {total_model_loading_time:.4f}s")
logger.info(f"  - ControlNet: {controlnet_load_time:.4f}s")
logger.info(f"  - VAE: {vae_load_time:.4f}s")
logger.info(f"  - Transformer: {transformer_load_time:.4f}s")
logger.info(f"  - Text encoders: {text_encoder_load_time + text_encoder_2_load_time + text_encoder_3_load_time:.4f}s")
logger.info(f"  - Tokenizers: {tokenizers_load_time:.4f}s")
logger.info(f"  - Scheduler: {scheduler_load_time:.4f}s")
logger.info(f"Pipeline assembly: {pipeline_assembly_time:.4f}s")
logger.info(f"Generation: {generation_time:.4f}s")
logger.info(f"Saving: {save_time:.4f}s")

# === SUMMARY OF MAJOR PHASES ===
logger.info("=== TIMING SUMMARY ===")
logger.info(f"Imports: {total_import_time:.4f}s")
logger.info(f"Depth processing: {total_depth_processing_time:.4f}s")
logger.info(f"Model loading: {total_model_loading_time:.4f}s")
logger.info(f"Generation: {generation_time:.4f}s")
logger.info(f"Saving: {save_time:.4f}s")
logger.info(f"Total execution time: {total_execution_time:.4f}s")

# Verification - sum of major phases should equal total
calculated_total = total_import_time + total_depth_processing_time + total_model_loading_time + generation_time + save_time
logger.info(f"Calculated total: {calculated_total:.4f}s (difference: {abs(total_execution_time - calculated_total):.4f}s)")
