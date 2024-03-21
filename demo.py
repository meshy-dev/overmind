from diffusers import DDIMScheduler, UNet2DConditionModel
from diffusers import DiffusionPipeline
import time
import pysnooper
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

# ckpt_path = hf_hub_download(
#     repo_id="ashawkey/LGM", filename="model_fp16.safetensors"
# )

from overmind.api import load
from huggingface_hub import hf_hub_download

@pysnooper.snoop(relative_time=True)
def foo():
    pipeline = load(DiffusionPipeline.from_pretrained, "stabilityai/stable-diffusion-xl-refiner-1.0")
    pipeline.to('cuda')
    print('ok')

foo()
time.sleep(1000)
