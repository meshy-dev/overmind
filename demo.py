from diffusers import DDIMScheduler, UNet2DConditionModel
import logging
import torch
from overmind.api import load
import sys

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

model_key = "stabilityai/stable-diffusion-2-1"
# model = DDIMScheduler.from_pretrained(
#     model_key, subfolder="scheduler", torch_dtype=torch.float16
# )

# model = load(DDIMScheduler.from_pretrained,
#     model_key, subfolder="scheduler", torch_dtype=torch.float16
# )

model = load(UNet2DConditionModel.from_pretrained,
    "meshy/MVDream", subfolder="unet", torch_dtype=torch.float16,
)  # use mvdream's config for the diffusers' model

print('ok')
