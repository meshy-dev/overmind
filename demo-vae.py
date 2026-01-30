# -*- coding: utf-8 -*-

# -- stdlib --
# -- third party --
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.models import AutoencoderKL
import overmind.api
import pysnooper
import torch

# -- code --
overmind.api.monkey_patch_all()

@pysnooper.snoop(relative_time=True, depth=2, color=False)
def load_pipeline():
    vae = (lambda: AutoencoderKL.from_pretrained(
        "lemon2431/ChineseInkComicStrip_v10",
        subfolder="vae",
        torch_dtype=torch.float16,
    ))()
    controlnet_depth = (lambda: ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16,
        variant="fp16",
    ))()
    controlnet_edge = (lambda: ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_softedge",
        torch_dtype=torch.float16,
        variant="fp16",
    ))()

    pipeline = (lambda: StableDiffusionControlNetPipeline.from_pretrained(
        "lemon2431/ChineseInkComicStrip_v10",
        controlnet=[controlnet_edge, controlnet_depth],
        vae=vae,
        torch_dtype=torch.float16,
        safety_checker=None,
    ))()

    (lambda: pipeline.to('cuda'))()

load_pipeline()
