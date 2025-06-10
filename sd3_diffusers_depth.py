import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

logger.info("Starting HuggingFace login...")
login_start = time.time()
login(os.getenv("HF_TOKEN"))
login_time = time.time() - login_start
logger.info(f"HuggingFace login took {login_time:.4f} seconds")

device = "cuda"
logger.info(f"Using device: {device}")

# Set cache directory
cache_dir = "./cache"
model_repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"#"stabilityai/stable-diffusion-3.5-large"
controlnet_repo_id = "InstantX/SD3-Controlnet-Depth"#"stabilityai/stable-diffusion-3.5-large-controlnet-canny"
torch_dtype = torch.bfloat16

logger.info(f"Model repo: {model_repo_id}")
logger.info(f"ControlNet repo: {controlnet_repo_id}")
logger.info(f"Cache directory: {cache_dir}")
logger.info(f"Torch dtype: {torch_dtype}")

# download an image
logger.info("Starting image download...")
image_load_start = time.time()
image = load_image(
    "./inputs/square.png"
)
image_load_time = time.time() - image_load_start
logger.info(f"Image loading took {image_load_time:.4f} seconds")

logger.info("Converting image to numpy array...")
np_conversion_start = time.time()
np_image = np.array(image)
np_conversion_time = time.time() - np_conversion_start
logger.info(f"Numpy conversion took {np_conversion_time:.4f} seconds")



logger.info("Loading Depth estimator...")
depth_estimator_load_start = time.time()
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
depth_estimator_load_time = time.time() - depth_estimator_load_start
logger.info(f"Depth estimator loading took {depth_estimator_load_time:.4f} seconds")

logger.info("Loading Depth feature extractor...")
feature_extractor_load_start = time.time()
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
feature_extractor_load_time = time.time() - feature_extractor_load_start
logger.info(f"Depth feature extractor loading took {feature_extractor_load_time:.4f} seconds")

logger.info("Generating Depth map...")
depth_start = time.time()
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
depth_image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
depth_image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
depth_image.save("outputs/diffusers/diffusers_depth_control.png")
depth_time = time.time() - depth_start
logger.info(f"Depth map generation took {depth_time:.4f} seconds")


# load control net and stable diffusion v1-5
logger.info("Loading ControlNet model...")
controlnet_load_start = time.time()
controlnet = SD3ControlNetModel.from_pretrained(
    controlnet_repo_id, 
    torch_dtype=torch_dtype,
    cache_dir=cache_dir
)
controlnet_load_time = time.time() - controlnet_load_start
logger.info(f"ControlNet loading took {controlnet_load_time:.4f} seconds")

logger.info("Loading Stable Diffusion pipeline...")
pipeline_load_start = time.time()
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    model_repo_id, 
    controlnet=controlnet, 
    torch_dtype=torch_dtype,
    cache_dir=cache_dir
)
pipe.to("cuda")
pipeline_load_time = time.time() - pipeline_load_start
logger.info(f"Pipeline loading took {pipeline_load_time:.4f} seconds")

total_model_load_time = controlnet_load_time + pipeline_load_time
logger.info(f"Total model loading time: {total_model_load_time:.4f} seconds")

# generate image
logger.info("Starting image generation...")
generation_start = time.time()
#generator = torch.Generator(device="cuda").manual_seed(24)
prompt = "studio ghibli style cartoon"
image = pipe(prompt,
    control_image=depth_image,
    controlnet_conditioning_scale=0.85,
    #generator=generator,
    height=1024, 
    width=1024,
    #num_inference_steps=60,
).images[0]
generation_time = time.time() - generation_start
logger.info(f"Image generation took {generation_time:.4f} seconds")

logger.info("Saving generated image...")
save_start = time.time()
image.save("outputs/diffusers/diffusers_depth_output.png")
save_time = time.time() - save_start
logger.info(f"Image saving took {save_time:.4f} seconds")

# Log total execution time
total_execution_time = time.time() - total_import_start
logger.info(f"Total execution time: {total_execution_time:.4f} seconds")

# Summary of major phases
logger.info("=== TIMING SUMMARY ===")
logger.info(f"Imports: {total_import_time:.4f}s")
logger.info(f"Image loading & processing: {image_load_time + np_conversion_time + depth_time:.4f}s")
logger.info(f"Model loading: {total_model_load_time:.4f}s")
logger.info(f"Generation: {generation_time:.4f}s")
logger.info(f"Total: {total_execution_time:.4f}s")
