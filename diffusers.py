import numpy as np
import random
import os
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
from huggingface_hub import login
import cv2
from PIL import Image

# Set cache directories
# os.environ["HF_HOME"] = "./cache"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "hub"
# os.environ["TRANSFORMERS_CACHE"] = "transformers"
# os.environ["HF_DATASETS_CACHE"] = "datasets"

login(os.getenv("HF_TOKEN"))
device = "cuda"
# Set cache directory
cache_dir = "./cache"
model_repo_id = "stabilityai/stable-diffusion-3.5-large"
controlnet_repo_id = "stabilityai/stable-diffusion-3.5-large-controlnet-canny"
torch_dtype = torch.bfloat16

# download an image
image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)
np_image = np.array(image)

# get canny image
np_image = cv2.Canny(np_image, 100, 200)
np_image = np_image[:, :, None]
np_image = np.concatenate([np_image, np_image, np_image], axis=2)
canny_image = Image.fromarray(np_image)

# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained(
    controlnet_repo_id, 
    torch_dtype=torch_dtype,
    cache_dir=cache_dir
)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    model_repo_id, 
    controlnet=controlnet, 
    torch_dtype=torch_dtype,
    cache_dir=cache_dir
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# generate image
generator = torch.manual_seed(0)
image = pipe(
    "futuristic-looking woman",
    num_inference_steps=20,
    generator=generator,
    image=image,
    control_image=canny_image,
).images[0]

image.save("outputs/diffusers/generated_image.png")
