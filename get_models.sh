#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
source .env
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    exit 1
fi
pip install --cache-dir=.venv/pip-cache -r requirements.txt

git clone https://github.com/CompVis/depth-fm.git depthfm

apt update && apt install -y aria2

# download the models:
mkdir -p models
cd models

# sd3.5
aria2c -x 16 -s 16 -k 1M \
  --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/sd3.5_large.safetensors"
mv ffef7a279d9134626e6ce0d494fba84fc1c7e720b3c7df2d19a09dc3796d8f93 sd3.5_large.safetensors

#canny
aria2c -x 16 -s 16 -k 1M \
  --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets/resolve/main/sd3.5_large_controlnet_canny.safetensors"
mv 4bc5cf949f6501a4bd125c6c1190e8fba0f1471f5ce36e9ebd1114867abedd4c sd3.5_large_controlnet_canny.safetensors

#depth
aria2c -x 16 -s 16 -k 1M \
  --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets/resolve/main/sd3.5_large_controlnet_depth.safetensors"
mv d6ded6aa4f60eda74ae48a8fdc1a9fa11b36f05488975916f1ecd4834fddffd0 sd3.5_large_controlnet_depth.safetensors

#blur
aria2c -x 16 -s 16 -k 1M \
  --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets/resolve/main/sd3.5_large_controlnet_blur.safetensors"
mv 43d71c6f570d93e04a2d00711c530c74dcd7eef5d322efaab967c2d8296854dc sd3.5_large_controlnet_blur.safetensors

# models/clip_g.safetensors
aria2c -x 16 -s 16 -k 1M \
  --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/text_encoders/clip_g.safetensors"
mv ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4 clip_g.safetensors

#models/clip_l.safetensors
aria2c -x 16 -s 16 -k 1M \
  --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/text_encoders/clip_l.safetensors"
mv 660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd clip_l.safetensors

 
#models/t5xxl.safetensors
aria2c -x 16 -s 16 -k 1M \
  --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/text_encoders/t5xxl_fp16.safetensors"
mv 6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635 t5xxl.safetensors
## warn warn - is it okay to remove _fp16?

# vae
aria2c -x 16 -s 16 -k 1M \
  --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/vae/diffusion_pytorch_model.safetensors"
mv 8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc sd3_vae.safetensors

# depthfm
wget https://ommer-lab.com/files/depthfm/depthfm-v1.ckpt

