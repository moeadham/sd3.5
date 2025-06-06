#!/usr/bin/env python3
"""
SD3 Inference Script with batch processing support and image preprocessing.

Usage:
    # Single image generation
    python sd3_infer.py --prompt "a photo of a cat" --steps 40 --seed 42
    
    # With preprocessed controlnet image
    python sd3_infer.py --prompt "studio ghibli style" --controlnet_ckpt models/sd3.5_large_controlnet_canny.safetensors --controlnet_cond_image inputs/canny.png
    
    # With raw image preprocessing (Canny)
    python sd3_infer.py --prompt "studio ghibli style" --controlnet_ckpt models/sd3.5_large_controlnet_canny.safetensors --raw_image_input photo.jpg --preprocess_type canny
    
    # With raw image preprocessing (Depth) 
    python sd3_infer.py --prompt "studio ghibli style" --controlnet_ckpt models/sd3.5_large_controlnet_depth.safetensors --raw_image_input photo.jpg --preprocess_type depth --depthfm_model_path path/to/depthfm.pth
    
    # With custom preprocessing parameters
    python sd3_infer.py --prompt "anime style" --controlnet_ckpt models/sd3.5_large_controlnet_canny.safetensors --raw_image_input photo.jpg --preprocess_type canny --canny_low_threshold 50 --canny_high_threshold 150
    
    # Batch mode with JSON config
    python sd3_infer.py --batch batch_config.json
    
Example batch config JSON format:
[
    {
        "prompt": "a beautiful sunset over mountains",
        "output": "sunset.png",
        "seed": 42,
        "control_strength": 0.7
    },
    {
        "prompt": "studio ghibli style",
        "raw_image_input": "photo.jpg",
        "preprocess_type": "canny",
        "canny_low_threshold": 80,
        "canny_high_threshold": 180,
        "controlnet_ckpt": "models/sd3.5_large_controlnet_canny.safetensors",
        "output": "ghibli_canny.png",
        "control_strength": 0.8
    },
    {
        "prompt": "oil painting style",
        "raw_image_input": "photo.jpg",
        "preprocess_type": "depth",
        "depth_num_steps": 4,
        "depth_ensemble_size": 6,
        "depthfm_model_path": "models/depthfm.pth",
        "controlnet_ckpt": "models/sd3.5_large_controlnet_depth.safetensors",
        "output": "oil_depth.png",
        "control_strength": 0.9
    }
]

Parameters:
    --raw_image_input: Path to raw image to preprocess
    --preprocess_type: Type of preprocessing ('canny' or 'depth')
    --depthfm_model_path: Path to DepthFM model checkpoint (required for depth preprocessing)
    --control_strength: Strength of controlnet influence (0.0-1.0, default 1.0)
    
Preprocessing Parameters:
    --canny_low_threshold: Lower threshold for Canny edge detection (default: 100)
    --canny_high_threshold: Upper threshold for Canny edge detection (default: 200)
    --depth_num_steps: Number of denoising steps for DepthFM (default: 2)
    --depth_ensemble_size: Number of predictions to ensemble for DepthFM (default: 4)

NOTE: All batch requests must use the same model configuration (model, vae, controlnet).
Different image sizes, prompts, seeds, etc. are allowed within a batch.

Required model files in `models` folder:
- `clip_g.safetensors` (openclip bigG, same as SDXL)
- `clip_l.safetensors` (OpenAI CLIP-L, same as SDXL)
- `t5xxl.safetensors` (google T5-v1.1-XXL)
- `sd3_medium.safetensors` (or whichever main MMDiT model file)
- `sd3_vae.safetensors` (optional, holds the VAE separately if needed)

For depth preprocessing, install DepthFM from https://github.com/CompVis/depth-fm
"""

import datetime
import json
import math
import os
import pickle
import re
import time
import logging
import cv2

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info(f"Script execution started at: {datetime.datetime.now().isoformat()}")

# Time imports
import_start = time.time()
logger.info(f"Starting imports at: {datetime.datetime.now().isoformat()}")

import_time = time.time()
import fire
logger.info(f"  fire imported in {time.time() - import_time:.2f}s")

import_time = time.time()
import numpy as np
logger.info(f"  numpy imported in {time.time() - import_time:.2f}s")

import_time = time.time()
import sd3_impls
logger.info(f"  sd3_impls imported in {time.time() - import_time:.2f}s")

import_time = time.time()
import torch
logger.info(f"  torch imported in {time.time() - import_time:.2f}s")

import_time = time.time()
import torchvision.transforms.functional as F
logger.info(f"  torchvision.transforms.functional imported in {time.time() - import_time:.2f}s")

import_time = time.time()
from other_impls import SD3Tokenizer, SDClipModel, SDXLClipG, T5XXLModel
logger.info(f"  other_impls imports in {time.time() - import_time:.2f}s")

import_time = time.time()
from PIL import Image
logger.info(f"  PIL imported in {time.time() - import_time:.2f}s")

import_time = time.time()
from safetensors import safe_open
logger.info(f"  safetensors imported in {time.time() - import_time:.2f}s")

import_time = time.time()
from sd3_impls import (
    SDVAE,
    BaseModel,
    CFGDenoiser,
    SD3LatentFormat,
    SkipLayerCFGDenoiser,
)
logger.info(f"  sd3_impls specific imports in {time.time() - import_time:.2f}s")

import_time = time.time()
from tqdm import tqdm
logger.info(f"  tqdm imported in {time.time() - import_time:.2f}s")

logger.info(f"Total import time: {time.time() - import_start:.2f}s")

#################################################################################################
### Image preprocessing functions
#################################################################################################

def preprocess_canny(img, canny_low_threshold=100, canny_high_threshold=200):
    """Convert PIL image to Canny edge detection.
    
    Args:
        img: PIL Image
        canny_low_threshold: Lower threshold for edge detection (default: 100)
        canny_high_threshold: Upper threshold for edge detection (default: 200)
    """
    preprocess_start = time.time()
    logger.info(f"Preprocessing image with Canny edge detection (thresholds: {canny_low_threshold}-{canny_high_threshold})...")
    
    # Convert PIL to tensor then to numpy
    img_tensor = F.to_tensor(img)
    img_np = img_tensor.numpy()
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_np.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
    
    # Convert to uint8 for Canny
    img_gray = (img_gray * 255).astype('uint8')
    
    # Apply Canny edge detection
    edges = cv2.Canny(img_gray, canny_low_threshold, canny_high_threshold)
    
    # Convert back to PIL Image
    edges_pil = Image.fromarray(edges)
    
    # Convert to RGB (Canny outputs single channel)
    edges_rgb = edges_pil.convert('RGB')
    
    logger.info(f"Canny preprocessing completed in {time.time() - preprocess_start:.2f}s")
    return edges_rgb


def preprocess_depth(img, depthfm_model_path=None, depth_num_steps=2, depth_ensemble_size=4):
    """Convert PIL image to depth map using DepthFM.
    
    Args:
        img: PIL Image
        depthfm_model_path: Path to DepthFM model checkpoint
        depth_num_steps: Number of denoising steps (default: 2)
        depth_ensemble_size: Number of predictions to ensemble (default: 4)
    """
    preprocess_start = time.time()
    logger.info(f"Preprocessing image with DepthFM (steps: {depth_num_steps}, ensemble: {depth_ensemble_size})...")
    
    try:
        from depthfm.dfm import DepthFM
    except ImportError:
        raise ImportError(
            "DepthFM not found. Please install it from https://github.com/CompVis/depth-fm"
        )
    
    if not depthfm_model_path:
        raise ValueError("depthfm_model_path must be provided for depth preprocessing")
    
    # Initialize DepthFM model
    logger.info(f"Loading DepthFM model from {depthfm_model_path}")
    
    depthfm_model = DepthFM(ckpt_path=depthfm_model_path)
    depthfm_model.eval()
    
    # Convert PIL to tensor
    img_tensor = F.to_tensor(img).unsqueeze(0)  # Add batch dimension
    c, h, w = img_tensor.shape[1:]
    
    # Resize to 512x512 for DepthFM
    img_resized = F.interpolate(img_tensor, (512, 512), mode='bilinear', align_corners=False)
    
    # Generate depth map
    with torch.no_grad():
        depth = depthfm_model(img_resized, num_steps=depth_num_steps, ensemble_size=depth_ensemble_size)
    
    # Resize back to original dimensions
    depth = F.interpolate(depth, (h, w), mode='bilinear', align_corners=False)
    
    # Convert to PIL Image
    depth_np = depth.squeeze().cpu().numpy()
    
    # Normalize to 0-255 range
    depth_normalized = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255).astype('uint8')
    
    # Convert to PIL and then to RGB
    depth_pil = Image.fromarray(depth_normalized)
    depth_rgb = depth_pil.convert('RGB')
    
    logger.info(f"Depth preprocessing completed in {time.time() - preprocess_start:.2f}s")
    return depth_rgb


#################################################################################################
### Wrappers for model parts
#################################################################################################


def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    load_start = time.time()
    loaded_keys = 0
    skipped_keys = 0
    
    for key in ckpt.keys():
        model_key = key
        if remap is not None and key in remap:
            model_key = remap[key]
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        logger.debug(
                            f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        skipped_keys += 1
                        break
            if obj is None:
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device)
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                # print(f"K: {model_key}, O: {obj.shape} T: {tensor.shape}")
                if obj.shape != tensor.shape:
                    logger.warning(
                        f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}"
                    )
                obj.set_(tensor)
                loaded_keys += 1
            except Exception as e:
                logger.error(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e
    
    logger.info(f"  Loaded {loaded_keys} keys, skipped {skipped_keys} keys in {time.time() - load_start:.2f}s")


CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}


class ClipG:
    def __init__(self, model_folder: str, device: str = "cpu"):
        init_start = time.time()
        logger.info("Initializing ClipG...")
        
        file_open_start = time.time()
        with safe_open(
            f"{model_folder}/clip_g.safetensors", framework="pt", device="cpu"
        ) as f:
            logger.info(f"  Opened clip_g.safetensors in {time.time() - file_open_start:.2f}s")
            
            model_create_start = time.time()
            self.model = SDXLClipG(CLIPG_CONFIG, device=device, dtype=torch.float32)
            logger.info(f"  Created SDXLClipG model in {time.time() - model_create_start:.2f}s")
            
            load_start = time.time()
            load_into(f, self.model.transformer, "", device, torch.float32)
            logger.info(f"  Loaded weights in {time.time() - load_start:.2f}s")
        
        logger.info(f"ClipG initialized in {time.time() - init_start:.2f}s")


CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}


class ClipL:
    def __init__(self, model_folder: str):
        init_start = time.time()
        logger.info("Initializing ClipL...")
        
        file_open_start = time.time()
        with safe_open(
            f"{model_folder}/clip_l.safetensors", framework="pt", device="cpu"
        ) as f:
            logger.info(f"  Opened clip_l.safetensors in {time.time() - file_open_start:.2f}s")
            
            model_create_start = time.time()
            self.model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device="cpu",
                dtype=torch.float32,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            logger.info(f"  Created SDClipModel in {time.time() - model_create_start:.2f}s")
            
            load_start = time.time()
            load_into(f, self.model.transformer, "", "cpu", torch.float32)
            logger.info(f"  Loaded weights in {time.time() - load_start:.2f}s")
        
        logger.info(f"ClipL initialized in {time.time() - init_start:.2f}s")


T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}


class T5XXL:
    def __init__(self, model_folder: str, device: str = "cpu", dtype=torch.float32):
        init_start = time.time()
        logger.info("Initializing T5XXL...")
        
        file_open_start = time.time()
        with safe_open(
            f"{model_folder}/t5xxl.safetensors", framework="pt", device="cpu"
        ) as f:
            logger.info(f"  Opened t5xxl.safetensors in {time.time() - file_open_start:.2f}s")
            
            model_create_start = time.time()
            self.model = T5XXLModel(T5_CONFIG, device=device, dtype=dtype)
            logger.info(f"  Created T5XXLModel in {time.time() - model_create_start:.2f}s")
            
            load_start = time.time()
            load_into(f, self.model.transformer, "", device, dtype)
            logger.info(f"  Loaded weights in {time.time() - load_start:.2f}s")
        
        logger.info(f"T5XXL initialized in {time.time() - init_start:.2f}s")


CONTROLNET_MAP = {
    "time_text_embed.timestep_embedder.linear_1.bias": "t_embedder.mlp.0.bias",
    "time_text_embed.timestep_embedder.linear_1.weight": "t_embedder.mlp.0.weight",
    "time_text_embed.timestep_embedder.linear_2.bias": "t_embedder.mlp.2.bias",
    "time_text_embed.timestep_embedder.linear_2.weight": "t_embedder.mlp.2.weight",
    "pos_embed.proj.bias": "x_embedder.proj.bias",
    "pos_embed.proj.weight": "x_embedder.proj.weight",
    "time_text_embed.text_embedder.linear_1.bias": "y_embedder.mlp.0.bias",
    "time_text_embed.text_embedder.linear_1.weight": "y_embedder.mlp.0.weight",
    "time_text_embed.text_embedder.linear_2.bias": "y_embedder.mlp.2.bias",
    "time_text_embed.text_embedder.linear_2.weight": "y_embedder.mlp.2.weight",
}


class SD3:
    def __init__(
        self, model, shift, control_model_file=None, verbose=False, device="cpu"
    ):
        init_start = time.time()
        logger.info(f"Initializing SD3 from {os.path.basename(model)}...")

        # NOTE 8B ControlNets were trained with a slightly different forward pass and conditioning,
        # so this is a flag to enable that logic.
        self.using_8b_controlnet = False

        file_open_start = time.time()
        with safe_open(model, framework="pt", device="cpu") as f:
            logger.info(f"  Opened SD3 model file in {time.time() - file_open_start:.2f}s")
            
            control_model_ckpt = None
            if control_model_file is not None:
                control_open_start = time.time()
                control_model_ckpt = safe_open(
                    control_model_file, framework="pt", device=device
                )
                logger.info(f"  Opened control model file in {time.time() - control_open_start:.2f}s")
            
            model_create_start = time.time()
            self.model = BaseModel(
                shift=shift,
                file=f,
                prefix="model.diffusion_model.",
                device="cuda",
                dtype=torch.float16,
                control_model_ckpt=control_model_ckpt,
                verbose=verbose,
            ).eval()
            logger.info(f"  Created BaseModel in {time.time() - model_create_start:.2f}s")
            
            load_start = time.time()
            load_into(f, self.model, "model.", "cuda", torch.float16)
            logger.info(f"  Loaded SD3 weights in {time.time() - load_start:.2f}s")
            
        if control_model_file is not None:
            control_load_start = time.time()
            control_model_ckpt = safe_open(
                control_model_file, framework="pt", device=device
            )
            
            move_start = time.time()
            self.model.control_model = self.model.control_model.to(device)
            logger.info(f"  Moved control model to {device} in {time.time() - move_start:.2f}s")
            
            load_start = time.time()
            load_into(
                control_model_ckpt,
                self.model.control_model,
                "",
                device,
                dtype=torch.float16,
                remap=CONTROLNET_MAP,
            )
            logger.info(f"  Loaded control model weights in {time.time() - load_start:.2f}s")

            self.using_8b_controlnet = (
                self.model.control_model.y_embedder.mlp[0].in_features == 2048
            )
            self.model.control_model.using_8b_controlnet = self.using_8b_controlnet
            logger.info(f"  Control model setup completed in {time.time() - control_load_start:.2f}s")
        control_model_ckpt = None
        
        logger.info(f"SD3 initialized in {time.time() - init_start:.2f}s")


class VAE:
    def __init__(self, model, dtype: torch.dtype = torch.float16):
        init_start = time.time()
        logger.info(f"Initializing VAE from {os.path.basename(model)}...")
        
        file_open_start = time.time()
        with safe_open(model, framework="pt", device="cpu") as f:
            logger.info(f"  Opened VAE model file in {time.time() - file_open_start:.2f}s")
            
            model_create_start = time.time()
            self.model = SDVAE(device="cpu", dtype=dtype).eval().cpu()
            logger.info(f"  Created SDVAE model in {time.time() - model_create_start:.2f}s")
            
            prefix_check_start = time.time()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            logger.info(f"  Checked prefix in {time.time() - prefix_check_start:.2f}s")
            
            load_start = time.time()
            load_into(f, self.model, prefix, "cpu", dtype)
            logger.info(f"  Loaded VAE weights in {time.time() - load_start:.2f}s")
        
        logger.info(f"VAE initialized in {time.time() - init_start:.2f}s")


#################################################################################################
### Main inference logic
#################################################################################################


# Note: Sigma shift value, publicly released models use 3.0
SHIFT = 3.0
# Naturally, adjust to the width/height of the model you have
WIDTH = 1024
HEIGHT = 1024
# Pick your prompt
PROMPT = "a photo of a cat"
# Negative prompt
NEGATIVE_PROMPT = ""
# Most models prefer the range of 4-5, but still work well around 7
CFG_SCALE = 4.5
# Different models want different step counts but most will be good at 50, albeit that's slow to run
# sd3_medium is quite decent at 28 steps
STEPS = 40
# Seed
SEED = 23
# SEEDTYPE = "fixed"
SEEDTYPE = "rand"
# SEEDTYPE = "roll"
# Actual model file path
# MODEL = "models/sd3_medium.safetensors"
# MODEL = "models/sd3.5_large_turbo.safetensors"
MODEL = "models/sd3.5_large.safetensors"
# VAE model file path, or set None to use the same model file
VAEFile = None  # "models/sd3_vae.safetensors"
# Optional init image file path
INIT_IMAGE = None
# ControlNet
CONTROLNET_COND_IMAGE = None
# If init_image is given, this is the percentage of denoising steps to run (1.0 = full denoise, 0.0 = no denoise at all)
DENOISE = 0.8
# Output file path
OUTDIR = "outputs"
# SAMPLER
SAMPLER = "dpmpp_2m"
# MODEL FOLDER
MODEL_FOLDER = "models"


class SD3Inferencer:

    def __init__(self):
        self.verbose = False

    def print(self, txt):
        if self.verbose:
            print(txt)

    def load(
        self,
        model=MODEL,
        vae=VAEFile,
        shift=SHIFT,
        controlnet_ckpt=None,
        model_folder: str = MODEL_FOLDER,
        text_encoder_device: str = "cpu",
        verbose=False,
        load_tokenizers: bool = True,
    ):
        load_start = time.time()
        logger.info("Starting model loading...")
        
        self.verbose = verbose
        
        tokenizer_start = time.time()
        logger.info("Loading tokenizers...")
        # NOTE: if you need a reference impl for a high performance CLIP tokenizer instead of just using the HF transformers one,
        # check https://github.com/Stability-AI/StableSwarmUI/blob/master/src/Utils/CliplikeTokenizer.cs
        # (T5 tokenizer is different though)
        self.tokenizer = SD3Tokenizer()
        logger.info(f"SD3Tokenizer loaded in {time.time() - tokenizer_start:.2f}s")
        
        if load_tokenizers:
            t5_start = time.time()
            logger.info("Loading Google T5-v1-XXL...")
            self.t5xxl = T5XXL(model_folder, text_encoder_device, torch.float32)
            logger.info(f"T5XXL total load time: {time.time() - t5_start:.2f}s")
            
            clipl_start = time.time()
            logger.info("Loading OpenAI CLIP L...")
            self.clip_l = ClipL(model_folder)
            logger.info(f"CLIP-L total load time: {time.time() - clipl_start:.2f}s")
            
            clipg_start = time.time()
            logger.info("Loading OpenCLIP bigG...")
            self.clip_g = ClipG(model_folder, text_encoder_device)
            logger.info(f"CLIP-G total load time: {time.time() - clipg_start:.2f}s")
        
        sd3_start = time.time()
        logger.info(f"Loading SD3 model {os.path.basename(model)}...")
        self.sd3 = SD3(model, shift, controlnet_ckpt, verbose, "cuda")
        logger.info(f"SD3 total load time: {time.time() - sd3_start:.2f}s")
        
        vae_start = time.time()
        logger.info("Loading VAE model...")
        self.vae = VAE(vae or model)
        logger.info(f"VAE total load time: {time.time() - vae_start:.2f}s")
        
        logger.info(f"All models loaded in {time.time() - load_start:.2f}s")

    def get_empty_latent(self, batch_size, width, height, seed, device="cuda"):
        self.print("Prep an empty latent...")
        shape = (batch_size, 16, height // 8, width // 8)
        latents = torch.zeros(shape, device=device)
        for i in range(shape[0]):
            prng = torch.Generator(device=device).manual_seed(int(seed + i))
            latents[i] = torch.randn(shape[1:], generator=prng, device=device)
        return latents

    def get_sigmas(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    def get_noise(self, seed, latent):
        generator = torch.manual_seed(seed)
        self.print(
            f"dtype = {latent.dtype}, layout = {latent.layout}, device = {latent.device}"
        )
        return torch.randn(
            latent.size(),
            dtype=torch.float32,
            layout=latent.layout,
            generator=generator,
            device="cpu",
        ).to(latent.dtype)

    def get_cond(self, prompt):
        cond_start = time.time()
        logger.info(f"Encoding prompt: '{prompt[:50]}...'")
        
        tokenize_start = time.time()
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        logger.info(f"  Tokenized in {time.time() - tokenize_start:.2f}s")
        
        clipl_start = time.time()
        l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
        logger.info(f"  CLIP-L encoded in {time.time() - clipl_start:.2f}s")
        
        clipg_start = time.time()
        g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
        logger.info(f"  CLIP-G encoded in {time.time() - clipg_start:.2f}s")
        
        t5_start = time.time()
        t5_out, t5_pooled = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
        logger.info(f"  T5XXL encoded in {time.time() - t5_start:.2f}s")
        
        concat_start = time.time()
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        result = torch.cat([lg_out, t5_out], dim=-2), torch.cat(
            (l_pooled, g_pooled), dim=-1
        )
        logger.info(f"  Concatenated in {time.time() - concat_start:.2f}s")
        
        logger.info(f"Total prompt encoding time: {time.time() - cond_start:.2f}s")
        return result

    def max_denoise(self, sigmas):
        max_sigma = float(self.sd3.model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def fix_cond(self, cond):
        cond, pooled = (cond[0].half().cuda(), cond[1].half().cuda())
        return {"c_crossattn": cond, "y": pooled}

    def do_sampling(
        self,
        latent,
        seed,
        conditioning,
        neg_cond,
        steps,
        cfg_scale,
        sampler="dpmpp_2m",
        controlnet_cond=None,
        control_strength=1.0,
        denoise=1.0,
        skip_layer_config={},
    ) -> torch.Tensor:
        sampling_start = time.time()
        logger.info(f"Starting sampling with {steps} steps, sampler: {sampler}, cfg_scale: {cfg_scale}, control_strength: {control_strength}")
        
        prepare_start = time.time()
        latent = latent.half().cuda()
        self.sd3.model = self.sd3.model.cuda()
        logger.info(f"  Moved to CUDA in {time.time() - prepare_start:.2f}s")
        
        noise_start = time.time()
        noise = self.get_noise(seed, latent).cuda()
        logger.info(f"  Generated noise in {time.time() - noise_start:.2f}s")
        
        sigma_start = time.time()
        sigmas = self.get_sigmas(self.sd3.model.model_sampling, steps).cuda()
        sigmas = sigmas[int(steps * (1 - denoise)) :]
        logger.info(f"  Prepared sigmas in {time.time() - sigma_start:.2f}s")
        
        cond_start = time.time()
        conditioning = self.fix_cond(conditioning)
        neg_cond = self.fix_cond(neg_cond)
        extra_args = {
            "cond": conditioning,
            "uncond": neg_cond,
            "cond_scale": cfg_scale,
            "controlnet_cond": controlnet_cond,
            "control_strength": control_strength,
        }
        logger.info(f"  Fixed conditions in {time.time() - cond_start:.2f}s")
        
        noise_scale_start = time.time()
        noise_scaled = self.sd3.model.model_sampling.noise_scaling(
            sigmas[0], noise, latent, self.max_denoise(sigmas)
        )
        logger.info(f"  Scaled noise in {time.time() - noise_scale_start:.2f}s")
        
        denoise_start = time.time()
        sample_fn = getattr(sd3_impls, f"sample_{sampler}")
        denoiser = (
            SkipLayerCFGDenoiser
            if skip_layer_config.get("scale", 0) > 0
            else CFGDenoiser
        )
        logger.info(f"  Starting denoising loop...")
        latent = sample_fn(
            denoiser(self.sd3.model, steps, skip_layer_config),
            noise_scaled,
            sigmas,
            extra_args=extra_args,
        )
        logger.info(f"  Denoising completed in {time.time() - denoise_start:.2f}s")
        
        postprocess_start = time.time()
        latent = SD3LatentFormat().process_out(latent)
        self.sd3.model = self.sd3.model.cpu()
        logger.info(f"  Post-processed and moved to CPU in {time.time() - postprocess_start:.2f}s")
        
        logger.info(f"Total sampling time: {time.time() - sampling_start:.2f}s")
        return latent

    def vae_encode(
        self, image, using_2b_controlnet: bool = False, controlnet_type: int = 0
    ) -> torch.Tensor:
        encode_start = time.time()
        logger.info("Encoding image to latent...")
        
        preprocess_start = time.time()
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        image_torch = torch.from_numpy(batch_images).cuda()
        
        if using_2b_controlnet:
            image_torch = image_torch * 2.0 - 1.0
        elif controlnet_type == 1:  # canny
            image_torch = image_torch * 255 * 0.5 + 0.5
        else:
            image_torch = 2.0 * image_torch - 1.0
        logger.info(f"  Preprocessed image in {time.time() - preprocess_start:.2f}s")
        
        move_start = time.time()
        image_torch = image_torch.cuda()
        self.vae.model = self.vae.model.cuda()
        logger.info(f"  Moved to CUDA in {time.time() - move_start:.2f}s")
        
        encode_vae_start = time.time()
        latent = self.vae.model.encode(image_torch).cpu()
        logger.info(f"  VAE encoding in {time.time() - encode_vae_start:.2f}s")
        
        move_back_start = time.time()
        self.vae.model = self.vae.model.cpu()
        logger.info(f"  Moved back to CPU in {time.time() - move_back_start:.2f}s")
        
        logger.info(f"Total VAE encoding time: {time.time() - encode_start:.2f}s")
        return latent

    def vae_encode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.unsqueeze(0)
        latent = SD3LatentFormat().process_in(latent)
        return latent

    def vae_decode(self, latent) -> Image.Image:
        decode_start = time.time()
        logger.info("Decoding latent to image...")
        
        move_start = time.time()
        latent = latent.cuda()
        self.vae.model = self.vae.model.cuda()
        logger.info(f"  Moved to CUDA in {time.time() - move_start:.2f}s")
        
        decode_vae_start = time.time()
        image = self.vae.model.decode(latent)
        image = image.float()
        logger.info(f"  VAE decoding in {time.time() - decode_vae_start:.2f}s")
        
        move_back_start = time.time()
        self.vae.model = self.vae.model.cpu()
        logger.info(f"  Moved back to CPU in {time.time() - move_back_start:.2f}s")
        
        postprocess_start = time.time()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        out_image = Image.fromarray(decoded_np)
        logger.info(f"  Post-processed image in {time.time() - postprocess_start:.2f}s")
        
        logger.info(f"Total VAE decoding time: {time.time() - decode_start:.2f}s")
        return out_image

    def _image_to_latent(
        self,
        image,
        width,
        height,
        using_2b_controlnet: bool = False,
        controlnet_type: int = 0,
    ) -> torch.Tensor:
        image_data = Image.open(image)
        image_data = image_data.resize((width, height), Image.LANCZOS)
        latent = self.vae_encode(image_data, using_2b_controlnet, controlnet_type)
        latent = SD3LatentFormat().process_in(latent)
        return latent

    def gen_image(
        self,
        prompts=[PROMPT],
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        steps=STEPS,
        cfg_scale=CFG_SCALE,
        sampler=SAMPLER,
        seed=SEED,
        seed_type=SEEDTYPE,
        out_dir=OUTDIR,
        controlnet_cond_image=CONTROLNET_COND_IMAGE,
        control_strength=1.0,
        init_image=INIT_IMAGE,
        denoise=DENOISE,
        skip_layer_config={},
    ):
        gen_start = time.time()
        logger.info(f"Starting image generation - {len(prompts)} prompt(s), {width}x{height}, {steps} steps")
        
        # Prepare latents
        latent_start = time.time()
        controlnet_cond = None
        if init_image:
            logger.info(f"Loading init image: {init_image}")
            latent = self._image_to_latent(init_image, width, height)
        else:
            latent = self.get_empty_latent(1, width, height, seed, "cpu")
            latent = latent.cuda()
        logger.info(f"  Latent preparation time: {time.time() - latent_start:.2f}s")
        
        # Prepare controlnet condition
        if controlnet_cond_image:
            control_start = time.time()
            logger.info(f"Loading controlnet condition image: {controlnet_cond_image}")
            using_2b, control_type = False, 0
            if self.sd3.model.control_model is not None:
                using_2b = not self.sd3.using_8b_controlnet
                control_type = int(self.sd3.model.control_model.control_type.item())
            controlnet_cond = self._image_to_latent(
                controlnet_cond_image, width, height, using_2b, control_type
            )
            logger.info(f"  ControlNet condition preparation time: {time.time() - control_start:.2f}s")
        
        # Encode negative prompt
        neg_start = time.time()
        neg_cond = self.get_cond(negative_prompt)
        logger.info(f"  Negative prompt encoding time: {time.time() - neg_start:.2f}s")
        
        seed_num = None
        pbar = tqdm(enumerate(prompts), total=len(prompts), position=0, leave=True)
        
        for i, prompt in pbar:
            prompt_start = time.time()
            logger.info(f"\nProcessing prompt {i+1}/{len(prompts)}: '{prompt[:50]}...'")
            
            # Determine seed
            if seed_type == "roll":
                seed_num = seed if seed_num is None else seed_num + 1
            elif seed_type == "rand":
                seed_num = torch.randint(0, 100000, (1,)).item()
            else:  # fixed
                seed_num = seed
            logger.info(f"  Using seed: {seed_num}")
            
            # Encode prompt
            cond_start = time.time()
            conditioning = self.get_cond(prompt)
            logger.info(f"  Prompt encoding time: {time.time() - cond_start:.2f}s")
            
            # Do sampling
            sampled_latent = self.do_sampling(
                latent,
                seed_num,
                conditioning,
                neg_cond,
                steps,
                cfg_scale,
                sampler,
                controlnet_cond,
                control_strength,
                denoise if init_image else 1.0,
                skip_layer_config,
            )
            
            # Decode and save
            image = self.vae_decode(sampled_latent)
            
            save_start = time.time()
            save_path = os.path.join(out_dir, f"{i:06d}.png")
            logger.info(f"  Saving to {save_path}")
            image.save(save_path)
            logger.info(f"  Saved in {time.time() - save_start:.2f}s")
            
            logger.info(f"  Total time for prompt {i+1}: {time.time() - prompt_start:.2f}s")
        
        logger.info(f"\nTotal generation time for {len(prompts)} images: {time.time() - gen_start:.2f}s")
    
    def process_single_request(self, config):
        """Process a single request configuration without reloading models."""
        request_start = time.time()
        
        # Extract parameters with defaults
        prompt = config.get('prompt', PROMPT)
        negative_prompt = config.get('negative_prompt', NEGATIVE_PROMPT)
        width = config.get('width', WIDTH)
        height = config.get('height', HEIGHT)
        steps = config.get('steps', STEPS)
        cfg_scale = config.get('cfg_scale', CFG_SCALE)
        sampler = config.get('sampler', SAMPLER)
        seed = config.get('seed', SEED)
        seed_type = config.get('seed_type', SEEDTYPE)
        output_path = config.get('output')
        controlnet_cond_image = config.get('controlnet_cond_image', CONTROLNET_COND_IMAGE)
        control_strength = config.get('control_strength', 1.0)
        init_image = config.get('init_image', INIT_IMAGE)
        denoise = config.get('denoise', DENOISE)
        skip_layer_config = config.get('skip_layer_config', {})
        
        # New preprocessing parameters
        raw_image_input = config.get('raw_image_input')
        preprocess_type = config.get('preprocess_type')  # 'canny' or 'depth'
        depthfm_model_path = config.get('depthfm_model_path')
        canny_low_threshold = config.get('canny_low_threshold', 100)
        canny_high_threshold = config.get('canny_high_threshold', 200)
        depth_num_steps = config.get('depth_num_steps', 2)
        depth_ensemble_size = config.get('depth_ensemble_size', 4)
        
        logger.info(f"\nProcessing request: prompt='{prompt[:50]}...', output={output_path}")
        logger.info(f"  Size: {width}x{height}, Steps: {steps}, Seed: {seed}, Control strength: {control_strength}")
        
        # Handle raw image preprocessing if needed
        if raw_image_input and preprocess_type:
            preprocess_start = time.time()
            logger.info(f"  Preprocessing raw image: {raw_image_input} with {preprocess_type}")
            
            # Load raw image
            raw_img = Image.open(raw_image_input).convert("RGB")
            
            # Apply preprocessing
            if preprocess_type == 'canny':
                processed_img = preprocess_canny(raw_img, canny_low_threshold, canny_high_threshold)
                # Save preprocessed image to the same directory as output
                if output_path:
                    # Get the output directory and filename
                    output_dir = os.path.dirname(output_path)
                    output_base = os.path.basename(output_path)
                    output_name, output_ext = os.path.splitext(output_base)
                    # Create control image path
                    control_image_path = os.path.join(output_dir, f"{output_name}_control{output_ext}")
                else:
                    # Fallback to original behavior if no output path specified
                    control_image_path = raw_image_input.replace('.', f'_canny.')
                # Ensure directory exists
                if output_path and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                processed_img.save(control_image_path)
                logger.info(f"  Saved Canny edges to: {control_image_path}")
                # Use the preprocessed image as controlnet condition
                controlnet_cond_image = control_image_path
            elif preprocess_type == 'depth':
                processed_img = preprocess_depth(raw_img, depthfm_model_path, depth_num_steps, depth_ensemble_size)
                # Save preprocessed image to the same directory as output
                if output_path:
                    # Get the output directory and filename
                    output_dir = os.path.dirname(output_path)
                    output_base = os.path.basename(output_path)
                    output_name, output_ext = os.path.splitext(output_base)
                    # Create control image path
                    control_image_path = os.path.join(output_dir, f"{output_name}_control{output_ext}")
                else:
                    # Fallback to original behavior if no output path specified
                    control_image_path = raw_image_input.replace('.', f'_depth.')
                # Ensure directory exists
                if output_path and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                processed_img.save(control_image_path)
                logger.info(f"  Saved depth map to: {control_image_path}")
                # Use the preprocessed image as controlnet condition
                controlnet_cond_image = control_image_path
            else:
                logger.warning(f"  Unknown preprocess_type: {preprocess_type}")
            
            logger.info(f"  Preprocessing completed in {time.time() - preprocess_start:.2f}s")
        
        # Prepare latents
        latent_start = time.time()
        controlnet_cond = None
        if init_image:
            logger.info(f"  Loading init image: {init_image}")
            latent = self._image_to_latent(init_image, width, height)
        else:
            latent = self.get_empty_latent(1, width, height, seed, "cpu")
            latent = latent.cuda()
        logger.info(f"  Latent preparation: {time.time() - latent_start:.2f}s")
        
        # Prepare controlnet condition
        if controlnet_cond_image:
            control_start = time.time()
            logger.info(f"  Loading controlnet condition: {controlnet_cond_image}")
            using_2b, control_type = False, 0
            if self.sd3.model.control_model is not None:
                using_2b = not self.sd3.using_8b_controlnet
                control_type = int(self.sd3.model.control_model.control_type.item())
            controlnet_cond = self._image_to_latent(
                controlnet_cond_image, width, height, using_2b, control_type
            )
            logger.info(f"  ControlNet condition: {time.time() - control_start:.2f}s")
        
        # Encode prompts
        cond_start = time.time()
        conditioning = self.get_cond(prompt)
        neg_cond = self.get_cond(negative_prompt)
        logger.info(f"  Prompt encoding: {time.time() - cond_start:.2f}s")
        
        # Generate seed based on type
        if seed_type == "rand":
            seed = torch.randint(0, 100000, (1,)).item()
            logger.info(f"  Using random seed: {seed}")
        
        # Do sampling
        sampled_latent = self.do_sampling(
            latent,
            seed,
            conditioning,
            neg_cond,
            steps,
            cfg_scale,
            sampler,
            controlnet_cond,
            control_strength,
            denoise if init_image else 1.0,
            skip_layer_config,
        )
        
        # Decode and save
        image = self.vae_decode(sampled_latent)
        
        if output_path:
            save_start = time.time()
            # Ensure directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            image.save(output_path)
            logger.info(f"  Saved to: {output_path} in {time.time() - save_start:.2f}s")
        else:
            logger.warning("  No output path specified, image not saved")
        
        logger.info(f"  Request completed in {time.time() - request_start:.2f}s")
        return image


def validate_batch_configs(configs):
    """Validate that all batch configs use the same model configuration."""
    if not configs:
        raise ValueError("Batch config is empty")
    
    # Model-related fields that must be the same across all configs
    model_fields = ['model', 'vae', 'controlnet_ckpt', 'shift', 'skip_layer_cfg']
    
    # Get reference values from first config
    first_config = configs[0]
    ref_values = {field: first_config.get(field) for field in model_fields}
    
    # Check all configs have same model configuration
    for i, config in enumerate(configs[1:], 1):
        for field in model_fields:
            if config.get(field) != ref_values[field]:
                raise ValueError(
                    f"Config {i} has different {field} value. "
                    f"All batch requests must use the same model configuration."
                )
    
    return True


def process_batch(batch_file, **model_kwargs):
    """Process a batch of generation requests from a JSON file."""
    batch_start = time.time()
    logger.info(f"Loading batch configuration from: {batch_file}")
    
    # Load batch config
    try:
        with open(batch_file, 'r') as f:
            batch_configs = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load batch config: {e}")
    
    if not isinstance(batch_configs, list):
        raise ValueError("Batch config must be a JSON array")
    
    logger.info(f"Loaded {len(batch_configs)} requests from batch file")
    
    # Validate all configs use same model
    validate_batch_configs(batch_configs)
    
    # Use model configuration from first config (or defaults)
    first_config = batch_configs[0]
    model = first_config.get('model', model_kwargs.get('model', MODEL))
    vae = first_config.get('vae', model_kwargs.get('vae', VAEFile))
    shift = first_config.get('shift', model_kwargs.get('shift', SHIFT))
    controlnet_ckpt = first_config.get('controlnet_ckpt', model_kwargs.get('controlnet_ckpt'))
    model_folder = first_config.get('model_folder', model_kwargs.get('model_folder', MODEL_FOLDER))
    text_encoder_device = first_config.get('text_encoder_device', model_kwargs.get('text_encoder_device', 'cpu'))
    verbose = first_config.get('verbose', model_kwargs.get('verbose', False))
    skip_layer_cfg = first_config.get('skip_layer_cfg', model_kwargs.get('skip_layer_cfg', False))
    
    # Get config for the model
    config = CONFIGS.get(os.path.splitext(os.path.basename(model))[0], {})
    _shift = shift or config.get("shift", 3)
    
    if skip_layer_cfg:
        skip_layer_config = CONFIGS.get(
            os.path.splitext(os.path.basename(model))[0], {}
        ).get("skip_layer_config", {})
    else:
        skip_layer_config = {}
    
    # Initialize inferencer and load models once
    logger.info("Initializing SD3 inferencer for batch processing...")
    inferencer = SD3Inferencer()
    inferencer.load(
        model,
        vae,
        _shift,
        controlnet_ckpt,
        model_folder,
        text_encoder_device,
        verbose,
    )
    
    # Process each request
    successful = 0
    failed = 0
    
    for i, config in enumerate(batch_configs, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing request {i}/{len(batch_configs)}")
        
        try:
            # Add skip_layer_config if needed
            if skip_layer_cfg and 'skip_layer_config' not in config:
                config['skip_layer_config'] = skip_layer_config
            
            inferencer.process_single_request(config)
            successful += 1
            logger.info(f"✓ Request {i}/{len(batch_configs)} completed successfully")
        except Exception as e:
            logger.error(f"Failed to process request {i}: {e}")
            failed += 1
            logger.error(f"✗ Request {i}/{len(batch_configs)} failed: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Batch processing completed in {time.time() - batch_start:.2f}s")
    logger.info(f"Successful: {successful}, Failed: {failed}")
    
    return successful, failed


CONFIGS = {
    "sd3_medium": {
        "shift": 1.0,
        "steps": 50,
        "cfg": 5.0,
        "sampler": "dpmpp_2m",
    },
    "sd3.5_medium": {
        "shift": 3.0,
        "steps": 50,
        "cfg": 5.0,
        "sampler": "dpmpp_2m",
        "skip_layer_config": {
            "scale": 2.5,
            "start": 0.01,
            "end": 0.20,
            "layers": [7, 8, 9],
            "cfg": 4.0,
        },
    },
    "sd3.5_large": {
        "shift": 3.0,
        "steps": 40,
        "cfg": 4.5,
        "sampler": "dpmpp_2m",
    },
    "sd3.5_large_turbo": {"shift": 3.0, "cfg": 1.0, "steps": 4, "sampler": "euler"},
    "sd3.5_large_controlnet_blur": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
    "sd3.5_large_controlnet_canny": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
    "sd3.5_large_controlnet_depth": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
}


@torch.no_grad()
def main(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    model=MODEL,
    out_dir=OUTDIR,
    postfix=None,
    seed=SEED,
    seed_type=SEEDTYPE,
    sampler=None,
    steps=None,
    cfg=None,
    shift=None,
    width=WIDTH,
    height=HEIGHT,
    controlnet_ckpt=None,
    controlnet_cond_image=None,
    control_strength=1.0,
    raw_image_input=None,
    preprocess_type=None,
    depthfm_model_path=None,
    canny_low_threshold=100,
    canny_high_threshold=200,
    depth_num_steps=2,
    depth_ensemble_size=4,
    vae=VAEFile,
    init_image=INIT_IMAGE,
    denoise=DENOISE,
    skip_layer_cfg=False,
    verbose=False,
    model_folder=MODEL_FOLDER,
    text_encoder_device="cpu",
    batch=None,
    **kwargs,
):
    main_start = time.time()
    logger.info(f"Main function started at: {datetime.datetime.now().isoformat()}")
    
    assert not kwargs, f"Unknown arguments: {kwargs}"
    
    # Handle batch mode
    if batch:
        logger.info(f"Running in batch mode with config file: {batch}")
        successful, failed = process_batch(
            batch,
            model=model,
            vae=vae,
            shift=shift,
            controlnet_ckpt=controlnet_ckpt,
            model_folder=model_folder,
            text_encoder_device=text_encoder_device,
            verbose=verbose,
            skip_layer_cfg=skip_layer_cfg,
        )
        logger.info(f"Batch processing complete. Successful: {successful}, Failed: {failed}")
        logger.info(f"Total execution time: {time.time() - main_start:.2f}s")
        return

    config_start = time.time()
    config = CONFIGS.get(os.path.splitext(os.path.basename(model))[0], {})
    _shift = shift or config.get("shift", 3)
    _steps = steps or config.get("steps", 50)
    _cfg = cfg or config.get("cfg", 5)
    _sampler = sampler or config.get("sampler", "dpmpp_2m")

    if skip_layer_cfg:
        skip_layer_config = CONFIGS.get(
            os.path.splitext(os.path.basename(model))[0], {}
        ).get("skip_layer_config", {})
        cfg = skip_layer_config.get("cfg", cfg)
    else:
        skip_layer_config = {}

    if controlnet_ckpt is not None:
        controlnet_config = CONFIGS.get(
            os.path.splitext(os.path.basename(controlnet_ckpt))[0], {}
        )
        _shift = shift or controlnet_config.get("shift", shift)
        _steps = steps or controlnet_config.get("steps", steps)
        _cfg = cfg or controlnet_config.get("cfg", cfg)
        _sampler = sampler or controlnet_config.get("sampler", sampler)
    
    logger.info(f"Configuration loaded in {time.time() - config_start:.2f}s")
    logger.info(f"Model: {os.path.basename(model)}")
    logger.info(f"Steps: {_steps}, CFG: {_cfg}, Sampler: {_sampler}, Shift: {_shift}")

    inferencer_start = time.time()
    inferencer = SD3Inferencer()
    logger.info(f"SD3Inferencer created in {time.time() - inferencer_start:.2f}s")

    load_start = time.time()
    inferencer.load(
        model,
        vae,
        _shift,
        controlnet_ckpt,
        model_folder,
        text_encoder_device,
        verbose,
    )
    logger.info(f"Model loading completed in {time.time() - load_start:.2f}s")

    prompt_prep_start = time.time()
    if isinstance(prompt, str):
        if os.path.splitext(prompt)[-1] == ".txt":
            logger.info(f"Loading prompts from file: {prompt}")
            with open(prompt, "r") as f:
                prompts = [l.strip() for l in f.readlines()]
            logger.info(f"Loaded {len(prompts)} prompts from file")
        else:
            prompts = [prompt]
    logger.info(f"Prompt preparation in {time.time() - prompt_prep_start:.2f}s")
    
    # Prepare output directory first (before preprocessing)
    output_prep_start = time.time()
    sanitized_prompt = re.sub(r"[^\w\-\.]", "_", prompt)
    out_dir = os.path.join(
        out_dir,
        (
            os.path.splitext(os.path.basename(model))[0]
            + (
                "_" + os.path.splitext(os.path.basename(controlnet_ckpt))[0]
                if controlnet_ckpt is not None
                else ""
            )
        ),
        os.path.splitext(os.path.basename(sanitized_prompt))[0][:50]
        + (postfix or datetime.datetime.now().strftime("_%Y-%m-%dT%H-%M-%S")),
    )

    os.makedirs(out_dir, exist_ok=False)
    logger.info(f"Output directory created: {out_dir}")
    logger.info(f"Output preparation in {time.time() - output_prep_start:.2f}s")
    
    # Handle raw image preprocessing in single mode (after output dir is created)
    if raw_image_input and preprocess_type:
        preprocess_start = time.time()
        logger.info(f"Preprocessing raw image: {raw_image_input} with {preprocess_type}")
        
        # Load raw image
        raw_img = Image.open(raw_image_input).convert("RGB")
        
        # Apply preprocessing
        if preprocess_type == 'canny':
            processed_img = preprocess_canny(raw_img, canny_low_threshold, canny_high_threshold)
            # Save preprocessed image to output directory with _control suffix
            control_image_path = os.path.join(out_dir, "000000_control.png")
            processed_img.save(control_image_path)
            logger.info(f"Saved Canny edges to: {control_image_path}")
            # Use the preprocessed image as controlnet condition
            controlnet_cond_image = control_image_path
        elif preprocess_type == 'depth':
            processed_img = preprocess_depth(raw_img, depthfm_model_path, depth_num_steps, depth_ensemble_size)
            # Save preprocessed image to output directory with _control suffix
            control_image_path = os.path.join(out_dir, "000000_control.png")
            processed_img.save(control_image_path)
            logger.info(f"Saved depth map to: {control_image_path}")
            # Use the preprocessed image as controlnet condition
            controlnet_cond_image = control_image_path
        else:
            logger.warning(f"Unknown preprocess_type: {preprocess_type}")
        
        logger.info(f"Preprocessing completed in {time.time() - preprocess_start:.2f}s")

    generation_start = time.time()
    inferencer.gen_image(
        prompts,
        negative_prompt,
        width,
        height,
        _steps,
        _cfg,
        _sampler,
        seed,
        seed_type,
        out_dir,
        controlnet_cond_image,
        control_strength,
        init_image,
        denoise,
        skip_layer_config,
    )
    logger.info(f"Image generation completed in {time.time() - generation_start:.2f}s")
    
    logger.info(f"Total execution time: {time.time() - main_start:.2f}s")
    logger.info(f"Output saved to: {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)