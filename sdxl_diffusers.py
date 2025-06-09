# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch

import cv2
from PIL import Image
from huggingface_hub import login

import os

# Set cache directories
os.environ["HF_HOME"] = "./cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "hub"
os.environ["TRANSFORMERS_CACHE"] = "transformers"
os.environ["HF_DATASETS_CACHE"] = "datasets"
# Set cache directory
cache_dir = "./cache"

prompt = "studio ghibli style"
negative_prompt = "low quality, bad quality, sketches"

# download an image
image = load_image(
    "./inputs/square.png"
)

# initialize the models and pipeline
controlnet_conditioning_scale = 0.85  # recommended for good generalization
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16, cache_dir=cache_dir
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    controlnet=controlnet, 
    vae=vae, 
    torch_dtype=torch.float16,
    cache_dir=cache_dir
)
#pipe.enable_model_cpu_offload()
pipe.to("cuda")

# get canny image
image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
canny_image.save("outputs/diffusers/sdxl_diffusers_canny.png")
# generate image
image = pipe(
    prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image
).images[0]
image.save("outputs/diffusers/sdxl_diffusers_output.png")