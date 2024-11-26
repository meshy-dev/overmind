import overmind.api
overmind.api.monkey_patch_all()

import torch
from diffusers.models import AutoencoderKL

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
)

checkpoint = "lemon2431/ChineseInkComicStrip_v10"
vae = AutoencoderKL.from_pretrained(
    "lemon2431/ChineseInkComicStrip_v10",
    subfolder="vae",
    torch_dtype=torch.float16,
)
controlnet_tile = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile", torch_dtype=torch.float16
)
controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float16,
    variant="fp16",
)
controlnet_edge = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_softedge",
    torch_dtype=torch.float16,
    variant="fp16",
)

text2img_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    checkpoint,
    controlnet=[controlnet_edge, controlnet_depth],
    vae=vae,
    torch_dtype=torch.float16,
    safety_checker=None,
)
