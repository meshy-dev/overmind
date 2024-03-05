from diffusers import DDIMScheduler, UNet2DConditionModel
import logging
import torch
from overmind.api import load
import sys
import torch
from huggingface_hub import hf_hub_download
import safetensors.torch

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

# model_key = "stabilityai/stable-diffusion-2-1"
# model = DDIMScheduler.from_pretrained(
#     model_key, subfolder="scheduler", torch_dtype=torch.float16
# )

# model = load(DDIMScheduler.from_pretrained,
#     model_key, subfolder="scheduler", torch_dtype=torch.float16
# )

# model = load(UNet2DConditionModel.from_pretrained,
#     "meshy/MVDream", subfolder="unet", torch_dtype=torch.float16,
# )  # use mvdream's config for the diffusers' model

ckpt_path = hf_hub_download(
    repo_id="ashawkey/LGM", filename="model_fp16.safetensors"
)

from overmind.api import load
from huggingface_hub import hf_hub_download
import safetensors.torch
ckpt_path = hf_hub_download(
    repo_id="stabilityai/stable-diffusion-xl-refiner-1.0", filename="vae/diffusion_pytorch_model.safetensors"
)
assert ckpt_path is not None, "Failed to download the model checkpoint"
foo = load(safetensors.torch.load_file, ckpt_path)

print('ok')
